import requests
from bs4 import BeautifulSoup
from openai import OpenAI
import time
import pandas as pd
client = OpenAI(api_key='HERE SHOULD BE YOUR API TOKEN') 

# URL of the website with the soap
URL_TO_SCRAPE = 'https://www.countryliving.com/life/a27452412/best-dad-jokes/'
# TODO: The 'openai.my_api_key' option isn't read in the client API. You will need to pass it when you instantiate the client, e.g. 'OpenAI(my_api_key='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')'
# openai.my_api_key = 

# Send a GET request to the URL
response = requests.get(URL_TO_SCRAPE)

# Check if the request was successful (status code 200)
if response.status_code == 200:
    # Parse the HTML content using BeautifulSoup
    soup = BeautifulSoup(response.content, 'html.parser')
    target_div = soup.find('div', class_='article-body-content article-body standard-body-content css-woto36 ewisyje7')

    if target_div:
        # Find all ul elements within the div
        ul_elements = target_div.find_all('ul')

        # Initialize an empty list to store all the elements from the lists
        all_elements = []

        # Iterate through each ul element
        for ul in ul_elements:
            # Find all li elements within each ul
            li_elements = ul.find_all('li')

            # Extract the text content of each li element and add it to the list
            all_elements.extend([li.get_text(strip=True)[1:-1].replace('"', '') for li in li_elements])

        replies = []
        for item in all_elements:
            messages = [
                {"role": "system", "content": "You are an assistant that helps to define possible audiance request for given joke. You receive a joke from me and should guess suitable request from audience for this joke. What would the audience request for a specific joke look like? Compose the request in form request:request_text"},
                {"role": "user", "content": item}
            ]
        
            chat = client.chat.completions.create(model="gpt-3.5-turbo", messages=messages)
            reply = chat.choices[0].message.content
            replies.append(reply)
            print(f'reply for joke "{item}" has been saved\n')
            time.sleep(20)
        joke_requests = [reply.split('request:')[1].strip().replace('"', '') for reply in replies]

        df = pd.DataFrame({
            'Request': joke_requests,
            'Joke': all_elements,
        })

        # Specify the file name
        csv_file_name = 'dad_jokes_dataset.csv'

        # Writing to CSV file
        df.to_csv(csv_file_name, index=False)
    else:
        print("The <div> element with the specified class was not found.")
else:
    print("Failed to retrieve the webpage.")