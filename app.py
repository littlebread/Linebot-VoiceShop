import sys
import configparser
import tempfile
import requests
import json
import pandas as pd
from pprint import pp

# Azure OpenAI
import os
from openai import AzureOpenAI
import json

from flask import Flask, request, abort
from linebot.v3 import (
    WebhookHandler
)
from linebot.v3.exceptions import (
    InvalidSignatureError
)
from linebot.v3.webhooks import (
    MessageEvent,
    TextMessageContent, AudioMessageContent
)
from linebot.v3.messaging import (
    Configuration,
    ApiClient,
    MessagingApi,
    MessagingApiBlob,
    ReplyMessageRequest,
    TextMessage,
    AudioMessage,
)

#Config Parser
config = configparser.ConfigParser()
config.read('config.ini')
UPLOAD_FOLDER = "static"


# Azure OpenAI Key
openai_client = AzureOpenAI(
    api_key=config["AzureOpenAI"]["KEY"],
    api_version=config["AzureOpenAI"]["VERSION"],
    azure_endpoint=config["AzureOpenAI"]["BASE"],
)

# Load dataset
df = pd.read_csv("美而美.csv")


system_messages = [
  {"role": "system", "content": """你是小美，一個先進的早餐店 AI 客服機器人。
* 你的任務是協助客人搜尋商品，然後加入購物車，最後完成結帳
* 若客人要求查詢購物車內有什麼商品，請用表格條列商品、金額和總金額
* 請總是用台灣繁體中文回答用戶，說話親切但是精準(Be Concise)
* 有時候工具會回報錯誤，請根據錯誤訊息處理，可向使用者詢問更多資訊，或是跟客戶道歉
* 不要跟用戶閒聊與點餐無關的事情
* 若問題不在上述內容中，請回答不知道，請客人聯繫客服""" },
]

prompt_messages = system_messages


tools = [
    {
        "type": "function",
        "function": {
            "name": "show_menu",
            "description": "回傳菜單"
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_products",
            "description": "搜尋商品，請用表格呈現商品列表，包括商品編號、商品名稱、單價，以及是否可購買(根據 qty 是否大於0)",
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword": {
                        "type": "string",
                        "description": "商品關鍵字，不能是空白",
                    }
                },
                "required": ["keyword"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_cart",
            "description": "加入購物車",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_id": {
                        "type": "string",
                        "description": "商品編號",
                    },
                    "qty": {
                        "type": "integer",
                        "description": "數量",
                    }
                },
                "required": ["product_id", "qty"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_order",
            "description": "用戶結帳",
            "parameters": {
                "type": "object",
                "properties": {
                  "receiver_name": {
                        "type": "string",
                        "description": "收件人",
                    },
                  "address": {
                        "type": "string",
                        "description": "地址",
                    }
                },
                "required": ["receiver_name", "address"],
            },
        },
    }
]


app = Flask(__name__)

channel_access_token = config['Line']['CHANNEL_ACCESS_TOKEN']
channel_secret = config['Line']['CHANNEL_SECRET']
if channel_secret is None:
    print('Specify LINE_CHANNEL_SECRET as environment variable.')
    sys.exit(1)
if channel_access_token is None:
    print('Specify LINE_CHANNEL_ACCESS_TOKEN as environment variable.')
    sys.exit(1)

handler = WebhookHandler(channel_secret)

configuration = Configuration(
    access_token=channel_access_token
)



@app.route("/callback", methods=['POST'])
def callback():
    # get X-Line-Signature header value
    signature = request.headers['X-Line-Signature']
    # get request body as text
    body = request.get_data(as_text=True)
    app.logger.info("Request body: " + body)

    # parse webhook body
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)
    return 'OK'

@handler.add(MessageEvent, message=TextMessageContent)
def message_text(event):
    with ApiClient(configuration) as api_client:
        line_bot_api = MessagingApi(api_client)
        line_bot_api.reply_message_with_http_info(
            ReplyMessageRequest(
                reply_token=event.reply_token,
                messages=[TextMessage(text=event.message.text)]
            )
        )

# Audio Message Type
@handler.add(
    MessageEvent,
    message=(AudioMessageContent),
)
def handle_content_message(event):
    with ApiClient(configuration) as api_client:
        line_bot_blob_api = MessagingApiBlob(api_client)
        message_content = line_bot_blob_api.get_message_content(
            message_id=event.message.id
        )
        with tempfile.NamedTemporaryFile(
            dir=UPLOAD_FOLDER, prefix="", delete=False
        ) as tf:
            tf.write(message_content)
            tempfile_path = tf.name

    original_file_name = os.path.basename(tempfile_path)
    # os.rename(
    #     UPLOAD_FOLDER + "/" + original_file_name,
    #     UPLOAD_FOLDER + "/" + "output.m4a",
    # )
    try:
        os.rename(UPLOAD_FOLDER + "/" + original_file_name, 
                  UPLOAD_FOLDER + "/" + "output.m4a")
    except FileExistsError:
        os.remove(UPLOAD_FOLDER + "/" + "output.m4a")
        os.rename(UPLOAD_FOLDER + "/" + original_file_name, 
                  UPLOAD_FOLDER + "/" + "output.m4a")

    with ApiClient(configuration) as api_client:
        whisper_result = azure_whisper()
        prompt_messages.append( { "role": "user", "content": whisper_result })
        result = get_completion_with_function_execution(prompt_messages, model=model_name, tools=tools)
        prompt_messages.append( { "role": "assistant", "content": result })

        line_bot_api = MessagingApi(api_client)
        line_bot_api.reply_message(
            ReplyMessageRequest(
                reply_token=event.reply_token,
                messages=[
                    TextMessage(text=whisper_result),
                    TextMessage(text=result)
                ],
            )
        )

def azure_whisper():
    audio_file = open("static/output.m4a", "rb")
    transcript = openai_client.audio.transcriptions.create(
        model=config["AzureOpenAI"]["WHISPER_DEPLOYMENT_NAME"], file=audio_file
    )
    audio_file.close()
    return transcript.text


model_name = config["AzureOpenAI"]["MODEL_NAME"]

def get_completion(messages, model= config["AzureOpenAI"]["MODEL_NAME"], temperature=0, max_tokens=4000, tools=None, tool_choice=None):
  payload = { "model": model, "temperature": temperature, "messages": messages, "max_tokens": max_tokens }

  if tools:
    payload["tools"] = tools
  if tool_choice:
    payload["tool_choice"] = tool_choice

  print('發送payload:')
  pp(payload)

  headers = { "api-key": f'{config["AzureOpenAI"]["KEY"]}', "Content-Type": "application/json" }
  response = requests.post(f'{config["AzureOpenAI"]["BASE"]}/openai/deployments/{model}/chat/completions?api-version={config["AzureOpenAI"]["VERSION"]}', headers = headers, data = json.dumps(payload) )
  obj = json.loads(response.text)
  print('LLM回應:')
  pp(obj)

  if response.status_code == 200 :
    return obj["choices"][0]["message"]
  else :
    return obj["error"]


def show_menu():
  print( f"called show_menu" )
  results = [{
    'product_id': row["product_id"],
    'title': row['title'],
    'qty': row["qty"],
    'price': row["price"]
  } for index, row in df.iterrows()]

  # 只回傳前5筆結果
  return results[:5]

def search_products(keyword):
  print( f"called search_products: {keyword}" )
  # 搜索包含關鍵詞的產品
  filtered_df = df[df['title'].str.contains(keyword, case=False, na=False)]

  results = [{
    'product_id': row["product_id"],
    'title': row['title'],
    'qty': row["qty"],
    'price': row["price"]
  } for index, row in filtered_df.iterrows()]

  # 只回傳前5筆結果
  return results[:5]

def add_cart(product_id, qty):
  print( f"called add_cart: {product_id} x {qty}" )
  if product_id in df['product_id'].values:
      # 獲取產品的數量
      product_qty = df.loc[df['product_id'] == product_id, 'qty'].iloc[0]
      if qty <= product_qty:
          return {"msg": "商品成功加入購物車"}
      else:
          # 缺貨
          return {"msg": "加入失敗，商品數量不足"}
  else:
      # 產品ID不存在
      return {"msg": "找不到此商品"}

def send_order(receiver_name, address):
  print( f"called send_order: {receiver_name} {address}")
  return "OK，訂單編號是 123456789"


available_functions = {
  "show_menu": show_menu,
  "search_products": search_products,
  "add_cart": add_cart,
  "send_order": send_order
}

def get_completion_with_function_execution(messages, model=config["AzureOpenAI"]["MODEL_NAME"], temperature=0, max_tokens=1000, tools=None, tool_choice=None):
  print(f"called prompt:")
  
  response = get_completion(messages, model=model, temperature=temperature, max_tokens=max_tokens, tools=tools,tool_choice=tool_choice)
  

  if response.get("tool_calls"): # 或用 response 裡面的 finish_reason 判斷也行
    messages.append(response)

    # ------ 呼叫函數，這裡改成執行多 tool_calls (可以改成平行處理，目前用簡單的迴圈)
    for tool_call in response["tool_calls"]:
      function_name = tool_call["function"]["name"]
      function_args = json.loads(tool_call["function"]["arguments"])
      function_to_call = available_functions[function_name]

      print(f"   called function {function_name} with {function_args}")
      function_response = function_to_call(**function_args)
      messages.append(
          {
              "tool_call_id": tool_call["id"], # 多了 toll_call_id
              "role": "tool",
              "name": function_name,
              "content": str(function_response),
          }
      )
      pp(messages)

    # 進行遞迴呼叫
    return get_completion_with_function_execution(messages, model=model, temperature=temperature, max_tokens=max_tokens, tools=tools,tool_choice=tool_choice)

  else:
    pp(response)
    return response["content"]








if __name__ == "__main__":
    app.run()