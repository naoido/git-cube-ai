from fastapi import FastAPI
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
import re
import requests
import base64

app = FastAPI()

@app.get("/todo")
async def get_prediction():
    


    GitHubName = "kenta-afk"
    GitHubRepo = "test"
    GitHubPath = "test.py"
    header_auth = {"Accept": "application/vnd.github.v3+json"}

    git_url = f"https://api.github.com/repos/{GitHubName}/{GitHubRepo}/contents/{GitHubPath}"

    response = requests.get(git_url, headers=header_auth)

    if response.status_code == 200:
        res = response.json()
        content = res['content']
        data = base64.b64decode(content).decode('utf-8')
        context = f'''{data}'''
    
    else:
        print(f"Error: {response.status_code}")

    model = AutoModelForCausalLM.from_pretrained("cyberagent/calm3-22b-chat", device_map="auto", torch_dtype="auto")
    tokenizer = AutoTokenizer.from_pretrained("cyberagent/calm3-22b-chat")
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    messages = [
    {"role": "system", "content": "あなたはどんな言語でも、コードをレビューすることができるプログラミングの専門家です。"},
    {"role": "system", "content": "##todo以下にあるコメントを*todoとします"},
    {"role": "system", "content": "##todoが複数ある場合は*todo_1, *todo_2...のようにリストで保存してください"},
    {"role": "system", "content": "##todoがある一つだけある場合は、*todoにたいしてどんな方法で実装するのが良いのかを30文字程度で考えて*ansとする。##todoが複数がある場合は、それぞれの*todoに対しての内容をどんな方法で実装するのが良いのかを30文字程度で考えて*ans_1, *ans_2...とする"},
    {"role": "system", "content": "*todoの内容が不鮮明の時は、コード全体を見て何をしようとしているか作成してください"},
    {"role": "system", "content": "[ [ { todo : *todo }, { ans : *ans } ] ]で返信してください。 *todoが複数ある場合は[ [ { todo : *todo_1 }, {ans : *ans_1 } ], [ { todo: *todo_2 }, { ans : *ans_2 } ] ]みたいに返信してください。"},
    {"role": "user", "content": context}
    ]

    def generate_response():
        input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
        output_ids = model.generate(input_ids, max_new_tokens=1024, temperature=0.5, streamer=streamer)
        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return generated_text



    text = generate_response()

    comp = text[862:]



    pattern = re.compile(r'"todo":\s*"([^"]*)".*?"ans":\s*"([^"]*)"', re.DOTALL)

    # 正規表現による抽出
    matches = pattern.findall(comp)

    prediction_result = {}

    # 結果の表示
    for match in matches:
        todo, ans = match
        print(f'todo: {todo}')
        
        prediction_result[todo] = ans
            
        return prediction_result



@app.get("/memo")
async def get_prediction(context):
    
    model = AutoModelForCausalLM.from_pretrained("cyberagent/calm3-22b-chat", device_map="auto", torch_dtype="auto")
    tokenizer = AutoTokenizer.from_pretrained("cyberagent/calm3-22b-chat")
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

  

    messages = [
        {"role": "system", "content": "あなたはいろいろな人のコメントに感想を返す人です。"},
        {"role": "system", "content": "コメントに対して感想を返してください。"},
        {"role": "user", "content": context}
    ]
    
    def generate_response():
        input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
        output_ids = model.generate(input_ids, max_new_tokens=1024, temperature=0.5, streamer=streamer)
        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return generated_text





    text = generate_response()
    thoughts = text[300:]
    return thoughts

    