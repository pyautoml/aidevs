#!/usr/bin/python3.11

__created__ = "04.04.2024"
__last_update__ = ""
__author__ = "https://github.com/pyautoml"
__course__ = "https://bravecourses.circle.so"

import gc
import json
import openai
import requests
from typing import Any


class OpenAiConnector:
    def __init__(self, settings_file: dict) -> None:
        try:
            self.settings = self.load_json(settings_file)
            self._api_key = self.settings["openai_api_key"]
            self._organization_id = self.settings["openai_organization_id"]
            self._headers = {
                "Authorization": f"Bearer {self._api_key}",
                "OpenAI-Organization": f"{self._organization_id}",
            }
            self._client = openai.OpenAI(api_key=self._api_key)
        except Exception as e:
            raise Exception(f"Error while loading settings: {e}")

    def load_json(self, json_file: str) -> dict:
        try:
            with open(json_file, "r") as file:
                return json.load(file)
        except Exception as e:
            raise Exception(f"Error processing for JSON file {json_file}: {e}")

    def generate_embedding(
        self,
        message: str,
        api_version: str = "v1",
        metadata: bool = False,
        embeddings_endpoint: str = None,
        embedding_type: str = "text-embedding-ada-002",
    ) -> Any:
        if not message:
            raise ValueError("Message cannot be empty.", flush=True)

        if not embeddings_endpoint:
            embeddings_endpoint = f"https://api.openai.com/{api_version}/embeddings"

        data = {"input": message, "model": embedding_type}

        try:
            response = requests.post(
                embeddings_endpoint, headers=self._headers, json=data
            ).json()
            try:
                if metadata:
                    return response["data"]
                else:
                    return response["data"][0]["embedding"]
            except KeyError as e:
                raise KeyError(f"Missing keys in response: {response.text}", flush=True)
        except Exception as e:
            raise Exception(f"Post error: {e}", flush=True)

    def prompt(
        self, prompt: str, model="gpt-4", max_tokens=300, temperature=0.1
    ) -> str:
        try:
            response = self._client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content
        except Exception as e:
            raise Exception(f"Error in OpenAi prompt: {e}")

    def prompt_vision(self, text: str, url: str, model: str = "gpt-4-vision-preview"):
        try:
            response = self._client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": text,
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": url,
                                },
                            },
                        ],
                    }
                ],
                max_tokens=330,
            )
            return response.choices[0].message.content
        except Exception as e:
            raise Exception(f"Error in vision prompt: {e}")


class Task:
    def __init__(self, settings_file: dict) -> None:
        try:
            self.settings = self.load_json(settings_file)
            self.openai = OpenAiConnector(settings_file=settings_file)
            self.task_api_key = self.settings["task_api_key"]
            self.task_data = self.url_validator(self.settings["task_data"])
            self.task_endpoint = self.url_validator(self.settings["task_endpoint"])
            self.task_token_endpoint = self.url_validator(
                self.settings["task_token_endpoint"]
            )
            self.task_answer_endpoint = self.url_validator(
                self.settings["task_answer_endpoint"]
            )
            self.form_headers = {"Content-Type": "application/x-www-form-urlencoded"}
            self.headers = {
                "Content-Type": "application/json",
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) Gecko/20100101 Firefox/124.0",
            }
        except KeyError as e:
            raise KeyError(f"Missing key in settings: {e}")
        except Exception as e:
            raise Exception(f"Error in loading data: {e}")

    def load_json(self, json_file: str) -> dict:
        try:
            with open(json_file, "r") as file:
                return json.load(file)
        except Exception as e:
            raise Exception(f"Error in reading JSON file: {e}")

    def speech_to_text(self, file_path: str, model: str = "whisper-1") -> str:
        try:
            audio_file = open(file_path, "rb")
            return self.openai._client.audio.transcriptions.create(
                model=model, file=audio_file
            ).text
        except Exception as e:
            raise Exception(f"Error in file transcription: {e}")

    def download_mp3_file(
        self,
        url: str = "https://tasks.aidevs.pl/data/mateusz.mp3",
        filename: str = "mateusz.mp3",
        directory: str = "files",
        return_file_path: str = False,
    ) -> str | None:
        mp3_directory = os.path.abspath(
            os.path.join(os.path.dirname(__file__), directory)
        )

        try:
            if not os.path.exists(mp3_directory):
                os.mkdir(mp3_directory, mode=0o777)
                print("Directory created")
        except Exception as e:
            raise Exception(f"Error while creating local directory: {e}")

        mp3_file = os.path.join(mp3_directory, filename)

        try:
            response = requests.get(url=url, stream=True)
        except Exception as e:
            raise Exception(f"MP3 request error: {e}")

        try:
            with open(mp3_file, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
            print("File downloaded successfully.")
            if return_file_path:
                return mp3_file
        except Exception as e:
            raise Exception(f"Error saving mp3 file: {e}")

    def url_validator(self, url: str) -> str:
        try:
            if not url.endswith("/"):
                url += "/"
            return url
        except Exception as e:
            raise Exception(f"Error in checking url: {e}")

    def get_task_token(self, task_name: str) -> str:
        try:
            url = self.task_token_endpoint + task_name
            payload = {"apikey": self.task_api_key}
            result = requests.post(url=url, json=payload)
            return result.json()["token"]
        except KeyError as e:
            raise KeyError(f"Missing 'token' key in JSON response: {e}")
        except Exception as e:
            raise Exception(f"Error retrieving task token: {e}")

    def get_task(self, token: str) -> dict:
        try:
            result = requests.get(url=self.task_endpoint + token)
            return result.json()
        except Exception as e:
            raise Exception(f"Error retrieving task token: {e}")

    def send_answer(self, token: str, payload: dict) -> dict:
        try:
            url = self.task_answer_endpoint + token
            result = requests.post(url=url, json=payload)
            return result.json()
        except Exception as e:
            raise Exception(f"Error while retrieving task token: {e}")

    # tasks ---
    def task_embedding(self, prompt: str = "Hawaiian pizza", printable: bool = False) -> dict|None:
        """
        Task name: embedding
        Task payload hint: https://zadania.aidevs.pl/hint/embedding
        """

        try:
            token = self.get_task_token(task_name="embedding")
            self.get_task(token=token)
            embedding = self.openai.generate_embedding(message=prompt)

            if not printable:
                return self.send_answer(token=token, payload={"answer": embedding})
            print(self.send_answer(token=token, payload={"answer": embedding}))
        except Exception as e:
            raise Exception(f"Error task 'embedding': {e}")

    def task_functions(self, printable: bool = False) -> dict|None:
        """
        Task name: functions
        """

        try:
            payload = {
                "name": "addUser",
                "description": "add new user",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "add first name"},
                        "surname": {"type": "string", "description": "add second name"},
                        "year": {"type": "integer", "description": "year of birth"},
                    },
                    "required": ["name", "surname", "year"],
                },
            }

            token = self.get_task_token(task_name="functions")
            self.get_task(token=token)

            if not printable:
                return self.send_answer(token=token, payload={"answer": payload})
            print(self.send_answer(token=token, payload={"answer": payload}))
        except Exception as e:
            raise Exception(f"Error task 'functions': {e}")

    def task_whoami(self, printable: bool = False, show_answer: bool = False) -> dict|None:
        """
        Task name: whoami
        Expected CLI output: Message: OK Status: CORRECT
        """
      
        try:
            token = self.get_task_token(task_name="whoami")
            task = self.get_task(token=token)
            prompt = f"Answer the question:'{task['hint']}' Knowing that: '{task['msg']}'. Return only answer without any comments."
            answer = self.openai.prompt(prompt=prompt)

            if show_answer:
                print(f"OpenAi answer: {answer}")
            if not printable:
                return self.send_answer(token=token, payload={"answer": answer})
            print(self.send_answer(token=token, payload={"answer": answer}))
        except KeyError as e:
            raise KeyError(f"Missing 'hint' or 'msg' key: {e}")
        except Exception as e:
            raise Exception(f"Error task 'whoami': {e}")

    def task_rodo(self, printable: bool = False) -> dict|None:
        """ Task name: rodo """
        
        try:
            token = self.get_task_token(task_name="rodo")
            self.get_task(token=token)
            answer = r"Use placeholders %imie%, %nazwisko%, %zawod%, %miasto% to replace confident data: name, surname, profession, city/town. Answer the question: Tell me everything about yourself. Change all names, cities, etc. for placeholders before returning answer."

            if not printable:
                return self.send_answer(token=token, payload={"answer": answer})
            print(self.send_answer(token=token, payload={"answer": answer}))
        except Exception as e:
            raise Exception(f"Error task 'rodo': {e}")

    def task_scraper(self, printable: bool = False, print_hints: bool = False, show_answer: bool = False) -> dict|None:
        """ Task name: scraper """
        try:
            status = None
            timeout = 10

            token = self.get_task_token(task_name="scraper")
            task_data = self.get_task(token=token)
            question = task_data["question"]
            url = task_data["input"]

            while status != 200:
                try:
                    result = requests.get(url=url, headers=self.headers, timeout=timeout)
                    status = result.status_code
                except KeyboardInterrupt:
                    exit()
                except Exception as e:
                    if str(e).endswith("Read timed out."):
                        timeout += randint(5,10)
                        if print_hints:
                            print("New timeout: ", timeout, flush=True)
                    else:
                        if print_hints:
                            print(f"Waiting reason: {e}\n")
                        timeout = 10
                        time.sleep(3)

            answer = self.openai.prompt(prompt=f"Return answer of max 200 characters for the question <QUESTION>{question}</QUESTION> in POLISH language, based on provided <DATA>{result.text}</DATA>")

            if show_answer:
                print(f"OpenAi answer: {answer}")
            if not printable:
                return self.send_answer(token=token, payload={"answer": answer})
            print(self.send_answer(token=token, payload={"answer": answer}))
        except Exception as e:
            raise Exception(f"Error task 'scraper': {e}")

 def task_liar(
        self,
        question: str = "Is the sky blue?",
        printable: bool = False,
        show_task: bool = False,
    ) -> dict|None:
        """ Task name: liar """

        token = self.get_task_token(task_name="liar")
        task = self.get_task(token=token)
        if show_task:
            print(f"Task instruction: {task}\n")
            
        result = requests.post(
            self.task_endpoint + token,
            headers=self.form_headers,
            data=f"question={question}",
        )
        aidevs_answer = result.json()["answer"]
        if show_task:
            print(f"Task text: {aidevs_answer}\n")

        prompt = f"Check if the text answers the question. Return only one value: YES or NO. <QUESTION>{question}</QUESTION> <TEXT>{aidevs_answer}</TEXT>"
        answer = self.openai.prompt(prompt=prompt)
        if show_task:
            print(f"OpenAi answer: {answer}\n")
            
        post_answer = self.send_answer(token=token, payload={"answer": answer})
        if printable:
            print(f"Task endpoint message: {post_answer}")
        else:
            return post_answer
            
    def task_whisper(
        self, printable: bool = False, show_task: bool = False
    ) -> dict | None:
        """ Task name: whisper """

        token = self.get_task_token(task_name="whisper")
        task_data = self.get_task(token=token)
        url = task_data["msg"].split(" ")[-1]

        if show_task:
            print(f"Url: {url}\n")

        if not url.startswith("http"):
            # then replace with a valid URL from 2024
            url = "https://tasks.aidevs.pl/data/mateusz.mp3"

        file_path = self.download_mp3_file(url=url, return_file_path=True)
        transcription = self.speech_to_text(file_path=file_path)

        if show_task:
            print(f"Transcription: {transcription}\n")

        answer = self.send_answer(token=token, payload={"answer": transcription})

        if printable:
            print(f"Answer: {answer}")
        else:
            return answer

    def task_search(
        self, 
        model: ollama, 
        percentage_treshold: int = 70,
        printable: bool = False, 
        directory: str = "files", 
        show_task: bool = False
    ) -> dict | None:
        """
        Task name: search
        Hint: Answer should be an URL address.
        """
        
        token = self.get_task_token(task_name="search")
        task_data = self.get_task(token=token)

        try:
            data_url = task_data["msg"]
            url = (data_url.split("-")[-1]).strip()
            question = task_data["question"]
        except KeyError as e:
            raise KeyError(f"Missing key 'data': {e}")
        
        try:
            response = requests.get(url=url, headers=self.headers)
            json_data = response.json()

            if not os.path.exists(f"{directory}/search.json"):
                with open(f"{directory}/search.json", "w", encoding="utf-8") as file:
                    json.dump(json_data, file, indent=4)
                    print("File saved successfully")
            else:
                print("File already exists")
        except Exception as e:
            raise Exception(f"Error while downloading JSON file: {e}")
        
        
        data = self.tools.load_custom_data(directory="./files/search.json")
        vectorized_question = model.create_embedding(text=question)["embedding"]
        embeddings = []
        titles = []
        urls = []
        top_3 = {}

        for row in data:
            title = row["title"]
            url = row["url"]
            embeddings.append(model.create_embedding(text=title)["embedding"])
            titles.append(title)
            urls.append(url)
        
        vectorized_question = np.array(len(data) * [vectorized_question])
        similarities = cosine_similarity(vectorized_question, np.array(embeddings))

        del embeddings
        del vectorized_question
        gc.collect()
        
        for i, similarity in enumerate(similarities[0], start=0):
            similarity_percentage = similarity * 100
            if similarity_percentage > percentage_treshold:
                top_3[urls[i]] = {
                    "similarity": f"{similarity_percentage:.2f}"
                }

        del similarities
        gc.collect()

        final_url = max(top_3, key=top_3.get)
        
        if show_task:
            print(f"Final url: {final_url}")
            print(f"Question: {question}")

        answer = self.send_answer(token=token, payload={"answer": final_url})

        if printable:
            print(answer)
        else:
            return answer     
            
def main():
    task = Task(settings_file="./configuration.json")
    open_ai = OpenAiConnector(settings_file="./configuration.json")
    model = OllamaModel(settings_file="./configuration.json")
   
    # task.task_embedding(printable=True)
    # task.task_functions(printable=True)
    # ttask.task_whoami(printable=True, show_answer=True)
    # task.task_rodo(printable=True)
    # task.task_scraper(printable=True, print_hints=True, show_answer=True)
    # task.task_liar(printable=True)
    # task.task_whisper(printable=True)
    # task.task_search(model=model, printable=True, show_task=True)

    del task
    del model
    del open_ai
    gc.collect()


if __name__ == "__main__":
    main()
  
