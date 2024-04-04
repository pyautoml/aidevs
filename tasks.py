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

    def send_answer(self, token: str, payload: dict) -> None:
        try:
            url = self.task_answer_endpoint + token
            result = requests.post(url=url, json=payload).json()
            print(f"Message: {result['msg']} Status: {result['note']}")
        except KeyError as e:
            raise KeyError(f"Missing 'msg' or 'note' keys in JSON response: {e}")
        except Exception as e:
            raise Exception(f"Error retrieving task token: {e}")

    # tasks ---
    def task_embedding(self, prompt: str = "Hawaiian pizza") -> None:
        """
        Task name: embedding
        Task payload hint: https://zadania.aidevs.pl/hint/embedding
        Expected CLI output: Message: OK Status: CORRECT
        """

        try:
            token = self.get_task_token(task_name="embedding")
            self.get_task(token=token)
            embedding = self.openai.generate_embedding(message=prompt)
            self.send_answer(token=token, payload={"answer": embedding})
        except Exception as e:
            raise Exception(f"Error task 'embedding': {e}")

    def task_functions(self) -> None:
        """
        Task name: functions
        Expected CLI output: Message: OK Status: CORRECT
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
            self.send_answer(token=token, payload={"answer": payload})
        except Exception as e:
            raise Exception(f"Error task 'functions': {e}")

    def task_whoami(self) -> None:
        """
        Task name: whoami
        Expected CLI output: Message: OK Status: CORRECT
        """
      
        try:
            token = self.get_task_token(task_name="whoami")
            task = self.get_task(token=token)
            prompt = f"Answer the question:'{task['hint']}' Knowing that: '{task['msg']}'. Return only answer without any comments."
            answer = self.openai.prompt(prompt=prompt)
            self.send_answer(token=token, payload={"answer": answer})
        except Exception as e:
            raise Exception(f"Error task 'whoami': {e}")


def main():
    task = Task(settings_file="./configuration.json")
    open_ai = OpenAiConnector(settings_file="./configuration.json")
   
    # task.task_embedding()
    # task.task_functions()
    # task.task_whoami()

    del task
    del open_ai
    gc.collect()


if __name__ == "__main__":
    main()
  
