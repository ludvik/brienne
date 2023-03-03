import os
import openai

openai.api_key=os.getenv("OPENAI_API_KEY")

session = []
session.append({"role": "system", "content": "You are a psycanalysis therapist. You will respond in Chinese. You have an assistant in the therapy room too. you can ask him for assistance, for.e.g. play music, etc. When you need assistance, place the command inside a [] quote."})

while True:
    raw_input = input(">>> ")
    session.append({"role": "user", "content": raw_input})

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=session
    )

    res_message = response["choices"][0]["message"]["content"]
    print("\n治疗师: ", res_message, "\n")
    session.append({"role": "assistant", "content": res_message})