# cognee/modules/search/llm_utils.py

import openai

# You can make this async, since get_completion uses `await`
async def generate_completion(query, context, user_prompt_path=None, system_prompt_path=None):
    prompt = f"Context: {context}\nQuestion: {query}\nAnswer:"
    
    try:
        response = await openai.ChatCompletion.acreate(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ]
        )
        return response.choices[0].message["content"]
    
    except Exception as e:
        return f"Error generating completion: {e}"
