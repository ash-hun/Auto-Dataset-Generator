import os
import tqdm
from openai import OpenAI

# GPT_Model, Last Edit : Nov 18th, 2023
GPT_MODEL = {
    '3.5':"gpt-3.5-turbo",
    '4':"gpt-4-1106-preview"
}

client = OpenAI(
    organization=os.getenv("OPENAI_ORGARNIZATION_KEY"), # 환경변수에 OPENAI_ORGANIZATION_KEY를 설정합니다.
    api_key=os.getenv("OPENAI_API_KEY") # 환경변수에 OPENAI_API_KEY를 설정합니다.
)

# Config
config = {
    'temperature': 1.0,
    'max_tokens':1000,
    'top_p':1.0,
    'best_of':1,
    'frequency_penalty':0.0,
    'presence_penalty':0.0
}

stop = ["\n"]

def generate(prompt_template:str, instruct_template:str, MODEL:str=GPT_MODEL['4']):
    """
    Generating QA Dataset Method

    Args:
        prompt_template (str): System Prompt
        instruct_template (str): Instruct Prompt 
        MODEL (str, optional): Usage AI Model in Open AI. Defaults to GPT_MODEL.

    Returns:
        str: msg response
    """

    completion = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": f"{prompt_template}"},
                {"role": "user", "content": f"{instruct_template}"}]
    )
    return completion.choices[0].message.content

if __name__ == "__main__":
    # ======================================================================
    # Generate Dataset without Chat history
    # ======================================================================
    bucket = []

    contents = [item for item in range(1, 11)] # Change your iterate variable or loop number
    with tqdm.tqdm(total=100) as pbar:
        for content in contents:
            state_message_history = []

            # Set System Prompt
            system_content = """
                your system prompt
            """

            # Set Instruct Prompt
            qa_gen = f"""
                your instruct prompt
            """
            
            qa_dataset = []
            
            res = generate(system_content, qa_gen)
            print(res) # logging
            bucket.append(res)
            
            pbar.update(100/len(contents)) # progress bar update

    with open('generate_data.txt', 'a+', encoding='utf-8') as t:
        for line in bucket:
            t.write(line + '\n')