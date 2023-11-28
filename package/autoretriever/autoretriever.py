import os
thisFile = os.path.realpath(__file__)
packageFolder = os.path.dirname(thisFile)
package = os.path.basename(packageFolder)
if os.getcwd() != packageFolder:
    os.chdir(packageFolder)
configFile = os.path.join(packageFolder, "config.py")
if not os.path.isfile(configFile):
    open(configFile, "a", encoding="utf-8").close()
from autoretriever import config

if not config.openaiApiKey:
    from autoretriever.health_check import HealthCheck
    HealthCheck.changeAPIkey()
    HealthCheck.saveConfig()
    print("Updated!")
HealthCheck.checkCompletion()

import autogen, os, json, traceback, chromadb, re
from pathlib import Path
from prompt_toolkit import prompt
from prompt_toolkit import print_formatted_text, HTML
from prompt_toolkit.styles import Style
from autogen.retrieve_utils import TEXT_FORMATS
from autogen.agentchat.contrib.retrieve_assistant_agent import RetrieveAssistantAgent
from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent


class AutoGenRetriever:

    def __init__(self):
        oai_config_list = []
        for model in ("gpt-3.5-turbo", "gpt-3.5-turbo-16k", "gpt-4", "gpt-4-32k"):
            oai_config_list.append({"model": model, "api_key": config.openaiApiKey})
        os.environ["OAI_CONFIG_LIST"] = json.dumps(oai_config_list)

    def getResponse(self, docs_path, message, auto=False):
        this_file = os.path.realpath(__file__)
        currentFolder = os.path.dirname(this_file)
        folder = os.path.join(currentFolder, "files")
        db = os.path.join(folder, "autogen", "retriever")
        Path(db).mkdir(parents=True, exist_ok=True)

        config_list = autogen.config_list_from_json(
            env_or_file="OAI_CONFIG_LIST",  # or OAI_CONFIG_LIST.json if file extension is added
            filter_dict={
                "model": {
                    config.chatGPTApiModel,
                }
            }
        )

        # https://microsoft.github.io/autogen/docs/reference/agentchat/contrib/retrieve_assistant_agent
        assistant = RetrieveAssistantAgent(
            name="assistant", 
            system_message="You are a helpful assistant.",
            llm_config={
                #"cache_seed": 42,  # seed for caching and reproducibility
                "config_list": config_list,  # a list of OpenAI API configurations
                "temperature": config.chatGPTApiTemperature,  # temperature for sampling
                "timeout": 600,
            },  # configuration for autogen's enhanced inference API which is compatible with OpenAI API
        )

        # https://microsoft.github.io/autogen/docs/reference/agentchat/contrib/retrieve_user_proxy_agent
        ragproxyagent = RetrieveUserProxyAgent(
            name="ragproxyagent",
            human_input_mode="NEVER" if auto else "ALWAYS",
            max_consecutive_auto_reply=config.max_consecutive_auto_reply,
            retrieve_config={
                #"task": "qa", # the task of the retrieve chat. Possible values are "code", "qa" and "default". System prompt will be different for different tasks. The default value is default, which supports both code and qa.
                "docs_path": docs_path,
                "chunk_token_size": 2000, # the chunk token size for the retrieve chat. If key not provided, a default size max_tokens * 0.4 will be used.
                "model": config_list[0]["model"],
                "client": chromadb.PersistentClient(path=db),
                "embedding_model": "all-mpnet-base-v2", # the embedding model to use for the retrieve chat. If key not provided, a default model all-MiniLM-L6-v2 will be used. All available models can be found at https://www.sbert.net/docs/pretrained_models.html. The default model is a fast model. If you want to use a high performance model, all-mpnet-base-v2 is recommended.
                "get_or_create": True,  # set to False if you don't want to reuse an existing collection, but you'll need to remove the collection manually
            },
        )

        ragproxyagent.initiate_chat(assistant, problem=message)

    def run(self):
        auto = False
        print("Do you want auto-reply (y/yes/N/NO)?")
        userInput = prompt(default="NO")
        if userInput.strip().lower() in ("y", "yes"):
            auto = True
            print("Enter maximum consecutive auto-reply below:")
            max_consecutive_auto_reply = prompt(default=str(config.max_consecutive_auto_reply),)
            try:
                if max_consecutive_auto_reply and int(max_consecutive_auto_reply) > 1:
                    config.max_consecutive_auto_reply = int(max_consecutive_auto_reply)
            except:
                print("Invalid entry!")

        print("AutoGen Retriever launched!")
        print("[press 'ctrl+q' to exit]")
        
        print("Enter your document path below (file / folder):")
        print(f"""Supported formats: *.{", *.".join(TEXT_FORMATS)}""")
        docs_path = prompt()

        # handle path dragged to terminal
        docs_path = docs_path.strip()
        docs_path = re.sub("^'(.*?)'$", r"\1", docs_path)
        if not os.path.isdir(docs_path) and "\\ " in docs_path:
            docs_path = docs_path.replace("\\ ", " ")

        if docs_path and os.path.exists(docs_path):
            print("Enter your query below:")
            message = prompt()
            if not message == config.exit_entry:
                try:
                    self.getResponse(docs_path, message, auto)
                except:
                    print(traceback.format_exc())
        else:
            print("Entered path does not exist!")
        
        print("\n\nAutoGen Retriever closed!")

def main():
    AutoGenRetriever().run()

if __name__ == '__main__':
    main()