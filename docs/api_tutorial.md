openlimit/gpt-5.4

kita akan pakai model itu, caranya gimana? Pakai curl

curl https://ai.sumopod.com/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" \
  -d '{
    "model": "gpt-4o-mini",
    "messages": [
      {
        "role": "user",
        "content": "Say hello in a creative way"
      }
    ],
    "max_tokens": 150,
    "temperature": 0.7
  }'

  atau

  from openai import OpenAI

client = OpenAI(
    api_key="sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
    base_url="https://ai.sumopod.com/v1"
)

# Stream the response
stream = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": "Write a short story about AI"}
    ],
    max_tokens=500,
    temperature=0.7,
    stream=True  # Enable streaming
)

# Print each chunk as it arrives
for chunk in stream:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="")

api key nya adalah sk-lwS0UngkX6W6W9ukB10wLw