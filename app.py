import numpy as np
import os
import re
import datetime
import time
import openai, tenacity
import argparse
import configparser
import json
import tiktoken
from get_paper_from_pdf import Paper
import gradio

# 定义Response类
class Response:
    # 初始化方法，设置属性
    def __init__(self, api, comment, language):
        self.api = api
        self.comment = comment
        self.language = language     
        self.max_token_num = 4096
        self.encoding = tiktoken.get_encoding("gpt2")
    
    
    @tenacity.retry(wait=tenacity.wait_exponential(multiplier=1, min=4, max=10),
                    stop=tenacity.stop_after_attempt(5),
                    reraise=True)
    def chat_response(self, comment):
        openai.api_key = self.api
        response_prompt_token = 1000        
        text_token = len(self.encoding.encode(comment))
        input_text_index = int(len(comment)*(self.max_token_num-response_prompt_token)/text_token)
        input_text = "This is the review comments:" + comment[:input_text_index]
        messages=[
                {"role": "system", "content": """You are the author, you submitted a paper, and the reviewers gave the review comments. 
                Please reply with what we have done, not what we will do.
                You need to extract questions from the review comments one by one, and then respond point-to-point to the reviewers’ concerns. 
                You need to determine for yourself how many reviewers there are and how many questions each reviewer has.
                Must be output in {}. Follow the format of the output later: 
                - Response to reviewers
                #1 reviewer
                Concern #1: xxxx
                Author response: xxxxx
                Concern #2: xxxx
                Author response: xxxxx
                ...
                #2 reviewer
                Concern #1: xxxx
                Author response: xxxxx
                Concern #2: xxxx
                Author response: xxxxx
                ...
                #3 reviewer
                Concern #1: xxxx
                Author response: xxxxx
                Concern #2: xxxx
                Author response: xxxxx
                ...
                
                """.format(self.language)
 
                },
                {"role": "user", "content": input_text},
            ]
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages,
            )
            result = ''
            for choice in response.choices:
                result += choice.message.content
            usage = response.usage.total_tokens
        except Exception as e:  
        # 处理其他的异常  
            result = "非常抱歉>_<，生了一个错误："+ str(e)
            usage  = 'xxxxx'
        print("********"*10)
        print(result)
        print("********"*10)                  
        return result, usage
                        
                           

def main(api, comment, language):  
    start_time = time.time()
    if not api or not comment:
        return "请输入API-key以及审稿意见！"
    else:
        Response1 = Response(api, comment, language)
        # 开始判断是路径还是文件：   
        response, total_token_used = Response1.chat_response(comment)
    time_used = time.time() - start_time
    output2 ="使用token数："+ str(total_token_used)+"\n花费时间："+ str(round(time_used, 2)) +"秒"
    return response, output2
        

########################################################################################################    
# 标题
title = "🤖ChatResponse🤖"
# 描述

description = '''<div align='left'>
<img align='right' src='http://i.imgtg.com/2023/03/22/94PLN.png' width="220">

<strong>ChatResponse是一款根据审稿人的评论自动生成作者回复的AI助手。</strong>其用途为：

⭐️根据输入的审稿意见，ChatResponse会自动提取其中各个审稿人的问题和担忧，并生成点对点的回复。

如果觉得很卡，可以点击右上角的Duplicate this Space，把ChatResponse复制到你自己的Space中！

本项目的[Github](https://github.com/nishiwen1214/ChatReviewer)，欢迎Star和Fork，也欢迎大佬赞助让本项目快速成长！💗

**很多人留言没有ChatGPT的API-key…不会申请API的可以加我微信"Shiwen_Ni"(备注api)**
</div>
'''

# 创建Gradio界面
inp = [gradio.inputs.Textbox(label="请输入你的API-key(sk开头的字符串)",
                          default="",
                          type='password'),
    gradio.inputs.Textbox(lines=5,
        label="请输入要回复的审稿意见",
        default=""
    ),
    gradio.inputs.Radio(choices=["English", "Chinese", "French", "German","Japenese"],
                        default="English",
                        label="选择输出语言"),
]

chat_Response_gui = gradio.Interface(fn=main,
                                 inputs=inp,
                                 outputs = [gradio.Textbox(lines=11, label="回复结果"), gradio.Textbox(lines=2, label="资源统计")],
                                 title=title,
                                 description=description)

# Start server
chat_Response_gui .launch(quiet=True, show_api=False)