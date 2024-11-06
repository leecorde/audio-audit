#program
import csv
from functools import partial
from groq import Groq
import glob
from io import BytesIO
import json
from operator import is_not
import pandas as pd
import re
import time
import tkinter as tk
from tkinter import filedialog
import requests
import streamlit as st
import sys
#WhisperX import
# import whisperx
import gc 
import torch
import os
from datetime import datetime
#LLM import
from langchain.embeddings import LlamaCppEmbeddings
from langchain_community.llms import LlamaCpp
from langchain_core.prompts import PromptTemplate
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from nltk.tree import Tree
from openpyxl import Workbook
from openpyxl.styles import PatternFill
from streamlit.components.v1 import html
import openai
from openai import OpenAI
from mutagen.mp3 import MP3
from mutagen.id3 import ID3, ID3NoHeaderError
from pydub import AudioSegment
import zipfile
import io
import httpx
import threading


#LLM Model and API key
#Specific the LLM Key for the API
#obtain from the LLM service providers the API key
#The LLM model can be selection or enter at the later step

#OpenAI model and API key
GPT_MODEL="gpt-3.5-turbo"
API_KEY = "sk-nyp-ai-devops-AgMUleAJ6EZILgFu5uaCT3BlbkFJldGad3ZgxG0hp2nhoGlJ"
client = OpenAI(api_key=API_KEY)


#Groq model and API key
GROQ_API_KEY="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" 
groq_client = Groq(api_key= GROQ_API_KEY)


#-------------------------------------------------------------------------
#Audio File Management
#save_audio_file() -to upload audio file to the server
#delete_mp3_files(directory) - to remove the mp3 files in a given directory

def save_audio_file(audio_bytes, name):
    try:
        if name.lower().endswith(".wav") or name.lower().endswith(".mp3"):
            file_name = "./" + f"audio_{name}"
            with open(file_name, "wb") as f:
                f.write(audio_bytes)
            print(f"File saved successfully: {file_name}")
            return file_name  # Ensure you return the file name
    except Exception as e:
        print(f"Failed to save file: {e}")
        return None  # Explicitly return None on failure

#delete_mp3_files - to delete the used audio file for audit
def delete_mp3_files(directory):
    # Construct the search pattern for MP3 files
    mp3_files = glob.glob(os.path.join(directory, "*.mp3"))
    
    for mp3_file in mp3_files:
        try:
            os.remove(mp3_file)
            # print(f"Deleted: {mp3_file}")
        except FileNotFoundError:
            print(f"{mp3_file} does not exist.")
        except Exception as e:
            print(f"Error deleting file {mp3_file}: {e}")


def convert_audio_to_wav(audio_file):
    audio = AudioSegment.from_file(audio_file)
    wav_file = audio_file.name.split(".")[0] + ".wav"
    audio.export(wav_file, format="wav")
    return wav_file

# def make_fetch_request(url, headers, method='GET', data=None):
#     if method == 'POST':
#         response = requests.post(url, headers=headers, json=data)
#     else:
#         response = requests.get(url, headers=headers)
#     return response.json()

#--------------------------------------------------------------
#Speech to text
#Function to convert the audio file into transcript(text)
# speech_to_text(audio file) - use openai whisper model to perform Speech to text
# speech_to_text_groq(audio_file) - use Groq whisper model to perform Speech to text
# The function receive the audio file will take 2 step to complete the transcript with speaker label
#Step 1: Whisper with audio file to convert the audio to text. The result is paragraph with all the text
#transcript
#Step 2: The text transcript will be pass to the LLM with prompt to instruct it to generate the transcript
#text with speaker label. The output will be dialog format between 2 speakers.

def speech_to_text_groq(audio_file):
    #print into dialog format
    dialog =""

    #Function to run Groq with user prompt
    #different model from Groq
    # Groq_model="llama3-8b-8192"
    # Groq_model="llama3-70b-8192"
    # Groq_model="mixtral-8x7b-32768"
    Groq_model="gemma2-9b-it"

    # Transcribe the audio
    audio_model="whisper-large-v3-turbo"

    with open(audio_file, "rb") as file:
        # Create a transcription of the audio file
        transcription = groq_client.audio.transcriptions.create(
        file=(audio_file, file.read()), # Required audio file
        model= audio_model, # Required model to use for transcription
        prompt="Elena Pryor, Samir, Sahil, Mihir, IPP, IPPFA",
        temperature=0,
        response_format="verbose_json"
          
        )

    # Print the transcription text
    print(transcription.text)
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
        {"role": "system", "content": """Insert speaker labels for a telemarketer and a customer. Return in a JSON format together with the original language code. Always translate the transcript fully to English."""},
        {"role": "user", "content": f"The audio transcript is: {transcription.text}"}
        ],
        temperature=0,
        max_tokens=16384
    )

    output = response.choices[0].message.content
    print(output)
    dialog = output.replace("json", "").replace("```", "")
    formatted_transcript = ""
    dialog = json.loads(dialog)
    language_code = dialog["language_code"]
    print(language_code)
    for entry in dialog['transcript']:
        formatted_transcript += f"{entry['speaker']}: {entry['text']}  \n\n"
    print(formatted_transcript)

    # Joining the formatted transcript into a single string
    dialog = formatted_transcript

    
    return dialog, language_code



def speech_to_text(audio_file):
    dialog =""

    # Transcribe the audio
    transcription = client.audio.transcriptions.create(
        model="whisper-1",
        file=open(audio_file, "rb"),
        prompt="Elena Pryor, Samir, Sahil, Mihir, IPP, IPPFA"

    )
    dialog = transcription.text
    # OPTIONAL: Uncomment the line below to print the transcription
    # print("Transcript: ", dialog + "  \n\n")

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
        {"role": "system", "content": """Insert speaker labels for a telemarketer and a customer. Return in a JSON format together with the original language code. Always translate the transcript fully to English."""},
        {"role": "user", "content": f"The audio transcript is: {dialog}"}
        ],
        temperature=0
    )

    output = response.choices[0].message.content
    # print(output)
    dialog = output.replace("json", "").replace("```", "")
    formatted_transcript = ""
    dialog = json.loads(dialog)
    language_code = dialog["language_code"]
    print(language_code)
    for entry in dialog['transcript']:
        formatted_transcript += f"{entry['speaker']}: {entry['text']}  \n\n"
    print(formatted_transcript)

    # Joining the formatted transcript into a single string
    dialog = formatted_transcript

   
    print(dialog)

    return dialog, language_code

#-----------------------------------------------------------------------------
#LLM Audit function
# LLM_audit(dialog) - use openai LLM for the audit generation
# groq_LLM_audit(dialog) - use Groq LLM for the audit generation
#
# There are 2 stages of the prompt in the function
#
# Stage1 + dialog prompt to test the criteria
# 1. Did the telemarketer introduced themselves by stating their name?
# 2. Did the telemarketer state that they are calling from 'IPPFA' or 'IPP Financial Advisors' without mentioning on behalf of any other insurers (Do not accept IPP)?            3. Did the customer asked how did the telemarketer obtained their contact details? If they asked, did telemarketer mentioned who gave the customer's details to him? (Not Applicable if customer didn't)
# 4. Did the telemarketer specify the types of financial services offered by IPP Financial Advisors or IPPFA?
# 5. Did the telemarketer offered to set up a meeting or zoom session with the consultant for the customer?
# 6. Did the telemarketer stated that products have high returns, guaranteed returns, or capital guarantee? (Fail if they did, Pass if they didn't)
# 7. Was the telemarketer polite and professional in their conduct?
#
# Stage2 + dialog prompt to test the criteria
# 1. Did the telemarketer ask if the customer is keen to explore how they can benefit from IPPFA's services?
# 2. Did the customer show uncertain response to the offer of the product and services? If Yes, Check did the telemarketer propose meeting or zoom session with company's consultant?
# 3. Did the telemarketer pressure the customer for the following activities (product introduction, setting an appointment)? (Fail if they did, Pass if they didn't)
# 
# Stage2 will test if all the criteria in Stage1 Pass
# All results are store in the Json format


def groq_LLM_audit(dialog):
    stage_1_prompt = """
    You are an auditor for IPP or IPPFA. 
    You are tasked with auditing a conversation between a telemarketer from IPP or IPPFA and a customer. 
    The audit evaluates whether the telemarketer adhered to specific criteria during the dialogue.

    ### Instruction:
        - Review the provided conversation transcript and assess the telemarketer's compliance based on the criteria outlined below. 
        - For each criterion, provide a detailed assessment, including quoting reasons from the conversation and a result status. 
        - Ensure all evaluations are based strictly on the content of the conversation. 
        - Only mark a criterion as "Pass" if you are very confident (i.e., nearly certain) based on clear and specific evidence from the conversation. 
        - Do not include words written in the brackets () as part of the criteria during the response.

        Audit Criteria:
            1. Did the telemarketer introduced themselves by stating their name?
            2. Did the telemarketer state that they are calling from 'IPPFA' or 'IPP Financial Advisors' without mentioning on behalf of any other insurers (Do not accept IPP)?            3. Did the customer asked how did the telemarketer obtained their contact details? If they asked, did telemarketer mentioned who gave the customer's details to him? (Not Applicable if customer didn't)
            4. Did the telemarketer specify the types of financial services offered by IPP Financial Advisors or IPPFA?
            5. Did the telemarketer offered to set up a meeting or zoom session with the consultant for the customer?
            6. Did the telemarketer stated that products have high returns, guaranteed returns, or capital guarantee? (Fail if they did, Pass if they didn't)
            7. Was the telemarketer polite and professional in their conduct?

        ** End of Criteria**

    ### Response:
        Generate JSON objects for each criteria in a list that must include the following keys:
        - "Criteria": State the criterion being evaluated.
        - "Reason": Provide specific reasons based on the conversation.
        - "Result": Indicate whether the criterion was met with "Pass", "Fail", or "Not Applicable".

        For Example:
            [
                {
                    "Criteria": "Did the telemarketer asked about the age of the customer",
                    "Reason": "The telemarketer asked how old the customer was.",
                    "Result": "Pass"
                }
            ]
    """ 

    stage_2_prompt = """
    You are an auditor for IPP or IPPFA. 
    You are tasked with auditing a conversation between a telemarketer from IPP or IPPFA and a customer. 
    The audit evaluates whether the telemarketer adhered to specific criteria during the dialogue.

    ### Instruction:
        - Review the provided conversation transcript and assess the telemarketer's compliance based on the criteria outlined below. 
        - For each criterion, provide a detailed assessment, including quoting reasons from the conversation and a result status. 
        - Ensure all evaluations are based strictly on the content of the conversation. 
        - Only mark a criterion as "Pass" if you are very confident (i.e., nearly certain) based on clear and specific evidence from the conversation. 
        - Do not include words written in the brackets () as part of the criteria during the response.

        Audit Criteria:
            1. Did the telemarketer ask if the customer is keen to explore how they can benefit from IPPFA's services?
            2. Did the customer show uncertain response to the offer of the product and services? If Yes, Check did the telemarketer propose meeting or zoom session with company's consultant?
            3. Did the telemarketer pressure the customer for the following activities (product introduction, setting an appointment)? (Fail if they did, Pass if they didn't)

        ** End of Criteria**

    ### Response:
        Generate JSON objects for each criteria in a list that must include the following keys:
        - "Criteria": State the criterion being evaluated.
        - "Reason": Provide specific reasons based on the conversation.
        - "Result": Indicate whether the criterion was met with "Pass", "Fail", or "Not Applicable".

        For Example:
            [
                {
                    "Criteria": "Did the telemarketer asked about the age of the customer",
                    "Reason": "The telemarketer asked how old the customer was.",
                    "Result": "Pass"
                }
            ]

    ### Input:
        %s
    """ % (dialog)

    chat_completion  = groq_client.chat.completions.create(
    model="llama3-groq-70b-8192-tool-use-preview",
    messages=[
        {
            "role": "system",
            "content": f"{stage_1_prompt}",
        },
        {
            "role": "user",
            "content": f"{dialog}",
        }
    ],
    temperature=0,
    max_tokens=4096,
    stream=False,
    stop=None,
    )
    stage_1_result = chat_completion.choices[0].message.content
    print(stage_1_result)
    

    stage_1_result = stage_1_result.replace("Audit Results:","")
    stage_1_result = stage_1_result.replace("### Input:","")
    stage_1_result = stage_1_result.replace("### Output:","")
    stage_1_result = stage_1_result.replace("### Response:","")
    stage_1_result = stage_1_result.replace("json","").replace("```","")
    stage_1_result = stage_1_result.strip()

    stage_1_result = json.loads(stage_1_result)

    print(stage_1_result)

    output_dict = {"Stage 1": stage_1_result}

    # for k,v in output_dict.items():
    #    person_names.append(get_person_entities(v[0]["Reason"]))

    #    if len(person_names) != 0:
            # print(person_names)
    #        v[0]["Result"] = "Pass"

    # print(output_dict)

    overall_result = "Pass"

    for i in range(len(stage_1_result)):
        if stage_1_result[i]["Result"] == "Fail":
            overall_result = "Fail"
            break  

    output_dict["Overall Result"] = overall_result

    if output_dict["Overall Result"] == "Pass":
        del output_dict["Overall Result"]

        chat_completion  = groq_client.chat.completions.create(
        model="llama3-groq-70b-8192-tool-use-preview",
        messages=[
            {
                "role": "system",
                "content": f"{stage_2_prompt}",
            },
            {
                "role": "user",
                "content": f"{dialog}",
            }
        ],
        temperature=0,
        max_tokens=4096,
        stream=False,
        stop=None,
        )
        stage_2_result = chat_completion.choices[0].message.content
        
        stage_2_result = stage_2_result.replace("Audit Results:","")
        stage_2_result = stage_2_result.replace("### Input:","")
        stage_2_result = stage_2_result.replace("### Output:","")
        stage_2_result = stage_2_result.replace("### Response:","")
        stage_2_result = stage_2_result.replace("json","").replace("```","")
        stage_2_result = stage_2_result.strip()

        # print(stage_2_result)

        stage_2_result = json.loads(stage_2_result)
        
        output_dict["Stage 2"] = stage_2_result

        overall_result = "Pass"

        for i in range(len(stage_2_result)):
            if stage_2_result[i]["Result"] == "Fail":
                overall_result = "Fail"
                break  
                
        output_dict["Overall Result"] = overall_result

    # print(output_dict)
    import gc; gc.collect(); torch.cuda.empty_cache();

    return output_dict
        

    



def LLM_audit(dialog):
    stage_1_prompt = """
    You are an auditor for IPP or IPPFA. 
    You are tasked with auditing a conversation between a telemarketer from IPP or IPPFA and a customer. 
    The audit evaluates whether the telemarketer adhered to specific criteria during the dialogue.

    ### Instruction:
        - Review the provided conversation transcript and assess the telemarketer's compliance based on the criteria outlined below. 
        - For each criterion, provide a detailed assessment, including quoting reasons from the conversation and a result status. 
        - Ensure all evaluations are based strictly on the content of the conversation. 
        - Only mark a criterion as "Pass" if you are very confident (i.e., nearly certain) based on clear and specific evidence from the conversation. 
        - Do not include words written in the brackets () as part of the criteria during the response.

        Audit Criteria:
            1. Did the telemarketer introduced themselves by stating their name?
            2. Did the telemarketer state that they are calling from 'IPPFA' or 'IPP Financial Advisors' without mentioning on behalf of any other insurers?
            3. Did the customer asked how did the telemarketer obtained their contact details? If they asked, did telemarketer mentioned who gave the customer's details to him? (Not Applicable if customer didn't)
            4. Did the telemarketer specify the types of financial services offered by IPP Financial Advisors or IPPFA?
            5. Did the telemarketer offered to set up a meeting or zoom session with the consultant for the customer?
            6. Did the telemarketer stated that products have high returns, guaranteed returns, or capital guarantee? (Fail if they did, Pass if they didn't)
            7. Was the telemarketer polite and professional in their conduct?

        ** End of Criteria**

    ### Response:
        Generate JSON objects for each criteria in a list that must include the following keys:
        - "Criteria": State the criterion being evaluated.
        - "Reason": Provide specific reasons based on the conversation.
        - "Result": Indicate whether the criterion was met with "Pass", "Fail", or "Not Applicable".

        For Example:
            [
                {
                    "Criteria": "Did the telemarketer asked about the age of the customer",
                    "Reason": "The telemarketer asked how old the customer was.",
                    "Result": "Pass"
                }
            ]

    ### Input:
        %s
    """ % (dialog)

    stage_2_prompt = """
    You are an auditor for IPP or IPPFA. 
    You are tasked with auditing a conversation between a telemarketer from IPP or IPPFA and a customer. 
    The audit evaluates whether the telemarketer adhered to specific criteria during the dialogue.

    ### Instruction:
        - Review the provided conversation transcript and assess the telemarketer's compliance based on the criteria outlined below. 
        - For each criterion, provide a detailed assessment, including quoting reasons from the conversation and a result status. 
        - Ensure all evaluations are based strictly on the content of the conversation. 
        - Only mark a criterion as "Pass" if you are very confident (i.e., nearly certain) based on clear and specific evidence from the conversation. 
        - Do not include words written in the brackets () as part of the criteria during the response.

        Audit Criteria:
            1. Did the telemarketer ask if the customer is keen to explore how they can benefit from IPPFA's services?
            2. Did the customer show uncertain response to the offer of the product and services? If Yes, Check did the telemarketer propose meeting or zoom session with company's consultant?
            3. Did the telemarketer pressure the customer for the following activities (product introduction, setting an appointment)? (Fail if they did, Pass if they didn't)

        ** End of Criteria**

    ### Response:
        Generate JSON objects for each criteria in a list that must include the following keys:
        - "Criteria": State the criterion being evaluated.
        - "Reason": Provide specific reasons based on the conversation.
        - "Result": Indicate whether the criterion was met with "Pass", "Fail", or "Not Applicable".

        For Example:
            [
                {
                    "Criteria": "Did the telemarketer asked about the age of the customer",
                    "Reason": "The telemarketer asked how old the customer was.",
                    "Result": "Pass"
                }
            ]

    ### Input:
        %s
    """ % (dialog)


    # Set up the model and prompt
    # model_engine = "text-davinci-003"
    model_engine ="gpt-4o-mini"

    messages=[{'role':'user', 'content':f"{stage_1_prompt}"}]


    completion = client.chat.completions.create(
    model=model_engine,
    messages=messages,
    temperature=0,)

    # print(completion)

    # extracting useful part of response
    stage_1_result = completion.choices[0].message.content
    stage_1_result = stage_1_result.replace("Audit Results:","")
    stage_1_result = stage_1_result.replace("### Input:","")
    stage_1_result = stage_1_result.replace("### Output:","")
    stage_1_result = stage_1_result.replace("### Response:","")
    stage_1_result = stage_1_result.replace("json","").replace("```","")
    stage_1_result = stage_1_result.strip()

    print(stage_1_result)

    def format_json_with_line_break(json_string):
        # Step 1: Add missing commas after "Criteria" and "Reason" key-value pairs
        corrected_json = re.sub(r'("Criteria":\s*".+?")(\s*")', r'\1,\2', json_string)
        corrected_json = re.sub(r'("Reason":\s*".+?")(\s*")', r'\1,\2', corrected_json)

        # Ensure there is a newline after the comma for "Criteria"
        corrected_json = re.sub(r'("Criteria":\s*".+?"),(\s*")', r'\1,\n\2', corrected_json)
        
        # Ensure there is a newline after the comma for "Reason"
        corrected_json = re.sub(r'("Reason":\s*".+?"),(\s*")', r'\1,\n\2', corrected_json)

        return corrected_json

    def get_person_entities(text):
        # Tokenize the text
        tokens = word_tokenize(text)
        
        # Apply part-of-speech tagging
        pos_tags = pos_tag(tokens)
        
        # Perform named entity recognition (NER)
        named_entities = ne_chunk(pos_tags)
        
        # Extract PERSON entities
        person_entities = []
        for chunk in named_entities:
            if isinstance(chunk, Tree) and chunk.label() == 'PERSON' or isinstance(chunk, Tree) and chunk.label() == 'FACILITY' or isinstance(chunk, Tree) and chunk.label() == 'GPE':
                person_name = ' '.join([token for token, pos in chunk.leaves()])
                person_entities.append(person_name)
        return person_entities

    person_names = []


    stage_1_result = format_json_with_line_break(stage_1_result)
    stage_1_result = json.loads(stage_1_result)

    output_dict = {"Stage 1": stage_1_result}

    # for k,v in output_dict.items():
    #    person_names.append(get_person_entities(v[0]["Reason"]))

    #    if len(person_names) != 0:
            # print(person_names)
    #        v[0]["Result"] = "Pass"

    # print(output_dict)

    overall_result = "Pass"

    for i in range(len(stage_1_result)):
        if stage_1_result[i]["Result"] == "Fail":
            overall_result = "Fail"
            break  

    output_dict["Overall Result"] = overall_result

    if output_dict["Overall Result"] == "Pass":
        del output_dict["Overall Result"]


        messages=[{'role':'user', 'content':f"{stage_2_prompt}"}]

        model_engine ="gpt-4o-mini"

        completion = client.chat.completions.create(
        model=model_engine,
        messages=messages,
        temperature=0,)

        # print(completion)

        # extracting useful part of response
        stage_2_result = completion.choices[0].message.content
        
        stage_2_result = stage_2_result.replace("Audit Results:","")
        stage_2_result = stage_2_result.replace("### Input:","")
        stage_2_result = stage_2_result.replace("### Output:","")
        stage_2_result = stage_2_result.replace("### Response:","")
        stage_2_result = stage_2_result.replace("json","").replace("```","")
        stage_2_result = stage_2_result.strip()

        print(stage_2_result)

        stage_2_result = format_json_with_line_break(stage_2_result)
        stage_2_result = json.loads(stage_2_result)
        
        output_dict["Stage 2"] = stage_2_result

        overall_result = "Pass"

        for i in range(len(stage_2_result)):
            if stage_2_result[i]["Result"] == "Fail":
                overall_result = "Fail"
                break  
                
        output_dict["Overall Result"] = overall_result

    print(output_dict)
    import gc; gc.collect(); torch.cuda.empty_cache();

    return output_dict
#--------------------------------------------------------------------
#Function to manage the folder
#For use if user want to copy audio file in a specific directory.
#Use this function to select the folder(direcory)
#Then the program will loop through the folder to perform audit for each audio file.



def select_folder():
   root = tk.Tk()
   root.wm_attributes('-topmost', 1)
   root.withdraw()
   folder_path = filedialog.askdirectory(parent=root)
    
   root.destroy()
   return folder_path

#------------------------------------------------------------------------------
#Function to create log files
#The log file will record each file that completed the audit or any error occured

def create_log_entry(event_description, log_file='logfile.txt', csv_file='logfile.csv'):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Log to text file
    with open(log_file, mode='a') as file:
        file.write(f"{timestamp} - {event_description}\n")
    
    # Log to CSV file
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        # Write header if the file is new
        if file.tell() == 0:
            writer.writerow(["timestamp", "event_description"])
        writer.writerow([timestamp, event_description])

#---------------------------------------------------------
#The Audit result can be download as a json format

@st.fragment
def handle_download_json(count, data, file_name, mime, log_message):
    st.download_button(
        label=f"Download {file_name.split('.')[-1].upper()}", 
        data=data,
        file_name=file_name,
        mime=mime,
        on_click=create_log_entry,
        args=(log_message,),
        key=f"download_json_{count}"
    )

#---------------------------------------------------------
#The Audit result can be download as a csv form

@st.fragment
def handle_download_csv(count, data, file_name, mime, log_message):
    st.download_button(
        label=f"Download {file_name.split('.')[-1].upper()}", 
        data=data,
        file_name=file_name,
        mime=mime,
        on_click=create_log_entry,
        args=(log_message,),
        key=f"download_csv_{count}"
    )

#---------------------------------------------------------
#download tje log file that contain the audio file names
#or error events
@st.fragment
def handle_download_log_file(data, file_name, mime, log_message):
    st.download_button(
        label="Download Logs", 
        data=data,
        file_name=file_name,
        mime=mime,
        on_click=create_log_entry,
        args=(log_message,),
    )

#---------------------------------------------------------------
#Download the text file contain the transcript of the conversion

@st.fragment
def handle_download_text(count, data, file_name, mime, log_message):
    st.download_button(
        label=f"Download Transcript", 
        data=data,
        file_name=file_name,
        mime=mime,
        on_click=create_log_entry,
        args=(log_message,),
        key=f"download_text_{count}"
    )

#----------------------------------------------------------------
#Down the result records in the zip file
#Zip contain:
#-All the transcripts
#-A csv file with all the audit result

@st.fragment
def zip_download(count, data, file_name, mime, log_message):
    st.download_button(
        label="Download All Files as ZIP",
        data=data,
        file_name=file_name,
        mime=mime,
        on_click=create_log_entry,
        args=(log_message,),
        key=f"download_zip_{count}"
    )

@st.fragment
def combined_audit_result_download(data, file_name, mime, log_message):
    st.download_button(
        label="Download Combined Audit Results As ZIP",
        data=data,
        file_name=file_name,
        mime=mime,
        on_click=create_log_entry,
        args=(log_message,),
    )


@st.fragment
def handle_combined_audit_result_download(data_text, data_csv, file_name_prefix):
    # Create an in-memory buffer for the ZIP file
    buffer = io.BytesIO()

    # Convert CSV data to a pandas DataFrame
    df = pd.read_csv(io.StringIO(data_csv))

    # Create an in-memory buffer for the Excel file (XLSX)
    xlsx_buffer = io.BytesIO()

    # Write DataFrame to XLSX format and add hyperlinks to the filenames
    with pd.ExcelWriter(xlsx_buffer, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Results')

        # Access the xlsxwriter workbook and worksheet objects
        # workbook = writer.book
        worksheet = writer.sheets['Results']

        # Add hyperlinks to text files based on the filenames in the DataFrame
        for index, row in df.iterrows():
            filename = row['Filename']  # Assuming 'filename' column exists
            # print(filename)
            if pd.notna(filename):  # Check if filename is not NaN (valid string)
                # Replace file extensions and create the hyperlink
                hyperlink = f"./{filename.replace('.mp3', '.txt').replace('.wav', '.txt')}"
                # Add the hyperlink to the 'filename' column in the Excel file (adjust the column index)
                worksheet.write_url(f"F{index + 2}", hyperlink, string=filename)

    # Move the pointer to the beginning of the xlsx_buffer to prepare it for reading
    xlsx_buffer.seek(0)

    with zipfile.ZipFile(buffer, "w") as zip_file:
        # Add text files
        for k, v in data_text.items():
            zip_file.writestr(k.replace(".mp3", ".txt").replace(".wav", ".txt"), v)

        # Add the CSV file as plain text
        # zip_file.writestr(f"{file_name_prefix}_file.csv", data_csv)

        # Add the XLSX file to the ZIP archive
        zip_file.writestr(f"{file_name_prefix}_file.xlsx", xlsx_buffer.read())

    # Move buffer pointer to the beginning of the ZIP buffer
    buffer.seek(0)

    # Return the buffer containing the ZIP archive
    return buffer

@st.fragment
def handle_combined_download(data_text, data_json, data_csv, file_name_prefix):
    # Create an in-memory buffer for the ZIP file
    buffer = io.BytesIO()

    with zipfile.ZipFile(buffer, "w") as zip_file:
        # Add text file
        zip_file.writestr(f"{file_name_prefix}_file.txt", data_text)

        # Add JSON file
        zip_file.writestr(f"{file_name_prefix}_file.json", data_json)

        # Add CSV file
        zip_file.writestr(f"{file_name_prefix}_file.csv", data_csv)

    # Move buffer pointer to the beginning
    buffer.seek(0)

    # Return the buffer containing the ZIP archive
    return buffer


def read_log_file(log_file='logfile.txt'):
    if os.path.exists(log_file):
        with open(log_file, 'r') as file:
            log_content = file.readlines()
        # Reverse the log entries
        log_content = log_content[::-1]
        if log_content:
            log_content[0] = f"<span style='color: yellow;'>{log_content[0].strip()}</span>\n"

        return ''.join(log_content)
    else:
        return "Log file does not exist."
    
    
def log_selection():
    method = st.session_state.upload_method
    if method == "Upload Files":
        create_log_entry("Method Chosen: File Upload")
    elif method == "Upload Folder":
        create_log_entry("Method Chosen: Folder Upload")

def is_valid_mp3(file_path):
    # Check if file exists
    if not os.path.isfile(file_path):
        print("File does not exist.")
        return False

    try:
        # Check the file using mutagen
        audio = MP3(file_path, ID3=ID3)
        
        # Check for basic properties
        if audio.info.length <= 0:  # Length should be greater than 0
            print("File is invalid: Length is zero or negative.")
            return False
        
        # You can check additional metadata if needed
        print("File is valid MP3 with duration:", audio.info.length)
        
        # Optional: Check if the file can be loaded with pydub
        AudioSegment.from_file(file_path)  # This will raise an exception if the file is not valid
        
        return True
    except (ID3NoHeaderError, Exception) as e:
        print(f"Invalid MP3 file: {e}")
        # create_log_entry(f"Error: Invalid MP3 file: {e}")
        return False

#Main flow that manage the program
#
#Side bar - to select the Speech To Text service provider
#
#Radio button to select upload method
#-upload the audio file for audit [ if method == "Upload Files":]
#-upload folder contain audio files[elif method == "Upload Folder":]
#
#Regardless what upload method is choose, the following step need to be performed
#Since this is the Server program, the client application need to upload the audio
#files to the server folder to be process[save_audio_file()]
#
# If submit is clicked [elif submit:]
# - Perform Speech To text  to produce a transcript [speech_to_text(audio_file)]
# - Use the transcript and LLM prompt to perform the 2 stages criteria test [LLM_audit(text)]
# - The result of the test for Json is converted to Dataframe to be saved as CSV.

#download the log file
#download the zip file(all transcript + result csv)

    

def main():

    with st.sidebar:
        st.header("AI Model Selection")
        transcribe_option = st.radio(
            "Choose your transcription AI model:",
            ("OpenAI (Recommended)", "Groq")
        )
        audit_option = st.radio(
            "Choose your Audit AI model:",
            ("OpenAI (Recommended)", "Groq")
        )
        st.write(f"Transcription Model:\n\n{transcribe_option.replace('(Recommended)','')}\n\nAudit Model:\n\n{audit_option.replace('(Recommended)','')}")
        st.markdown('<p style="color:red;">Groq AI Models are not recommended for file sizes of more than 1MB. Model will start to hallucinate.</p>', unsafe_allow_html=True)

    st.title("AI Transcription & Audit Service")
    st.info("Upload an audio file or select a folder to convert audio to text.", icon=":material/info:")

    audio_files = []
    status = ""

    # Initialize session state to track files and change detection
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = {}
    if 'file_change_detected' not in st.session_state:
        st.session_state.file_change_detected = False
    if 'audio_files' not in st.session_state:
        st.session_state.audio_files = []

    save_to_path = st.session_state.get("save_to_path", None)

    # Choose upload method
    method = st.radio("Select Upload Method:", options=["Upload Files", "Upload Folder"], horizontal=True, key='upload_method', on_change=log_selection)

    with st.expander("Other Options"):
        save_audited_transcript = st.checkbox("Save Audited Results to Folder (CSV)")
        if save_audited_transcript:
            save_to_button = st.button("Save To Folder")
            if save_to_button:
                save_to_path = select_folder()  # Assume this is a function that handles folder selection
                if save_to_path:
                    st.session_state.save_to_path = save_to_path
                    create_log_entry(f"Action: Save To Folder - {save_to_path}")

        save_to_display = st.empty()

        if save_audited_transcript == False:
            st.session_state.save_to_path = None
            save_to_path = None
            save_to_display.empty()
        else:
            save_to_display.write(f"Save To Folder: {save_to_path}")

    if method == "Upload Files":
        # File uploader
        uploaded_files = st.file_uploader(
            label="Choose audio files", 
            label_visibility="collapsed", 
            type=["wav", "mp3"], 
            accept_multiple_files=True
        )
        # # Create a set to track unique filenames
        # unique_filenames = set()

        # # Check for duplicates and collect unique filenames
        # for file in uploaded_files:
        #     if file.name in unique_filenames:
        #         del st.session_state['uploaded_files']
        #         st.warning("File has already been added!")
        #     else:
        #         unique_filenames.add(file.name)

        if uploaded_files is not None:
            # Track current files
            current_files = {file.name: file for file in uploaded_files}

            # Determine files that have been added
            added_files = [file_name for file_name in current_files if file_name not in st.session_state.uploaded_files]
            for file_name in added_files:
                create_log_entry(f"Action: File Uploaded - {file_name}")
                file = current_files[file_name]

                try:
                    audio_content = file.read()
                    saved_path = save_audio_file(audio_content, file_name)
                    
                    st.session_state.audio_files.append(saved_path)
                    st.session_state.file_change_detected = True


                #     if is_valid_mp3(saved_path):
                #         st.session_state.audio_files.append(saved_path)
                #         st.session_state.file_change_detected = True
                #     else:
                #         st.error(f"{saved_path[2:]} is an Invalid MP3 or WAV File")
                #         create_log_entry(f"Error: {saved_path[2:]} is an Invalid MP3 or WAV File")
                except Exception as e:
                    st.error(f"Error loading audio file: {e}")
                    create_log_entry(f"Error loading audio file: {e}")

            # Determine files that have been removed
            removed_files = [file_name for file_name in st.session_state.uploaded_files if file_name not in current_files]
            for file_name in removed_files:
                create_log_entry(f"Action: File Removed - {file_name}")
                st.session_state.audio_files = [f for f in st.session_state.audio_files if not f.endswith(file_name)]
                st.session_state.file_change_detected = True

            # Update session state with the current file list if a change was detected
            if st.session_state.file_change_detected:
                st.session_state.uploaded_files = current_files
                st.session_state.file_change_detected = False

        
        audio_files = list(st.session_state.audio_files)
        # print(st.session_state.audio_files)
        # print(type(audio_files))

    elif method == "Upload Folder":
        # create_log_entry("Method Chosen: Folder Upload")
        # Initialize the session state for folder_path
        selected_folder_path = st.session_state.get("folder_path", None)

        # Create two columns for buttons
        col1, col2 = st.columns(spec=[2, 8])

        with col1:
            # Button to trigger folder selection
            folder_select_button = st.button("Upload Folder")
            if folder_select_button:
                selected_folder_path = select_folder()  # Assume this is a function that handles folder selection
                if selected_folder_path:
                    st.session_state.folder_path = selected_folder_path
                    create_log_entry(f"Action: Folder Uploaded - {selected_folder_path}")

        with col2:
            # Option to remove the selected folder
            if selected_folder_path:
                remove_folder_button = st.button("Remove Uploaded Folder")
                if remove_folder_button:
                    st.session_state.folder_path = None
                    selected_folder_path = None
                    directory = "./"
                    delete_mp3_files(directory)
                    create_log_entry("Action: Uploaded Folder Removed")
                    success_message = "Uploaded folder has been removed."

        # Display the success message if it exists
        if 'success_message' in locals():
            st.success(success_message)

        # Display the selected folder path
        if selected_folder_path:
            st.write("Uploaded folder path:", selected_folder_path)

            # Get all files in the selected folder
            files_in_folder = os.listdir(selected_folder_path)
            st.write("Files in the folder:")

            # Process each file
            for file_name in files_in_folder:
                try:
                    file_path = os.path.join(selected_folder_path, file_name)
                    with open(file_path, 'rb') as file:
                        audio_content = file.read()
                        just_file_name = os.path.basename(file_name)
                        save_path = os.path.join(just_file_name)
                        saved_file_path = save_audio_file(audio_content, save_path)
                        audio_files.append(saved_file_path)
                        # if is_valid_mp3(saved_file_path):
                        #     audio_files.append(saved_file_path)
                        # else:
                        #     st.error(f"{saved_file_path[2:]} is an Invalid MP3 or WAV File")
                        #     create_log_entry(f"Error: {saved_file_path[2:]} is an Invalid MP3 or WAV File")


                except Exception as e:
                    # st.warning(f"Error processing file '{file_name}': {e}")
                    st.warning(f"Error processing file {file_name} is not an MP3 or WAV file.")
            
            #Filter files that are not in MP3 or WAV extensions
            audio_files = list(filter(partial(is_not, None), audio_files))

            st.write(audio_files)
            # print(audio_files)

    # Submit button
    submit = st.button("Submit", use_container_width=True)

    if submit and audio_files == []:
        create_log_entry("Service Request: Fail (No Files Uploaded)")
        st.error("No Files Uploaded, Please Try Again!")


    elif submit:
        combined_results = []
        all_text = {}
        if not save_audited_transcript or (save_audited_transcript and save_to_path != None):
            current = 1
            end = len(audio_files)
            for audio_file in audio_files:
                print(audio_file)
                if not os.path.isfile(audio_file):
                    st.error(f"{audio_file[2:]} Not Found, Please Try Again!")
                else:
                    try:
                        with st.spinner("Transcribing & Auditing In Progress..."):
                            if transcribe_option == "OpenAI (Recommended)":            
                                text, language_code = speech_to_text(audio_file)
                                if audit_option == "OpenAI (Recommended)":
                                    result = LLM_audit(text)
                                    if result["Overall Result"] == "Fail":
                                        status = "<span style='color: red;'> (FAIL)</span>"
                                    else:
                                        status = "<span style='color: green;'> (PASS)</span>"
                                elif audit_option == "Groq":
                                    result = groq_LLM_audit(text)
                                    if result["Overall Result"] == "Fail":
                                        status = "<span style='color: red;'> (FAIL)</span>"
                                    else:
                                        status = "<span style='color: green;'> (PASS)</span>"
                            elif transcribe_option == "Groq":
                                text, language_code = speech_to_text_groq(audio_file)
                                if audit_option == "OpenAI (Recommended)":
                                    result = LLM_audit(text)
                                    if result["Overall Result"] == "Fail":
                                        status = "<span style='color: red;'> (FAIL)</span>"
                                    else:
                                        status = "<span style='color: green;'> (PASS)</span>"
                                elif audit_option == "Groq":
                                    result = groq_LLM_audit(text)
                                    if result["Overall Result"] == "Fail":
                                        status = "<span style='color: red;'> (FAIL)</span>"
                                    else:
                                        status = "<span style='color: green;'> (PASS)</span>"
                    except Exception as e:
                        # Check if the error is related to CUDA memory
                        if "out of memory" in str(e):
                            import gc
                            import torch
                            
                            print("CUDA Test")
                            gc.collect()
                            torch.cuda.empty_cache()
                            try:
                                with st.spinner("Transcribing & Auditing In Progress..."):
                                    if transcribe_option == "OpenAI":
                                        text, language_code = speech_to_text(audio_file)
                                        if audit_option == "OpenAI":
                                            result = LLM_audit(text)
                                            if result["Overall Result"] == "Fail":
                                                status = "<span style='color: red;'> (FAIL)</span>"
                                            else:
                                                status = "<span style='color: green;'> (PASS)</span>"
                                        elif audit_option == "Groq":
                                            st.error("Under Maintainance")
                                    elif transcribe_option == "Groq":
                                        text, language_code = speech_to_text_groq(audio_file)
                                        if audit_option == "OpenAI":
                                            result = LLM_audit(text)
                                            if result["Overall Result"] == "Fail":
                                                status = "<span style='color: red;'> (FAIL)</span>"
                                            else:
                                                status = "<span style='color: green;'> (PASS)</span>"
                                        elif audit_option == "Groq":
                                            st.error("Under Maintainance")
                            except Exception as h:
                                create_log_entry(f"Error processing data: {h}")
                                st.error(f"Error processing data: {h}")
                        else:
                            create_log_entry(f"Error processing data: {e}")
                            st.error(f"Error processing data: {e}")
                    col1, col2 = st.columns([0.9,0.1])
                    with col1:
                        with st.expander(audio_file[2:] + f" ({language_code})"):
                        # with st.expander(audio_file[2:]):
                            st.write()
                            tab1, tab2, tab3 = st.tabs(["Converted Text", "Audit Result", "Download Content"])
                    with col2:
                        st.write(f"({current} / {end})")
                        st.markdown(status, unsafe_allow_html=True)
                    with tab1:
                        st.write(text)
                        all_text[audio_file[2:]] = text
                        print(all_text)
                        handle_download_text(count=audio_files.index(audio_file) ,data=text, file_name=f'{audio_file[2:].replace(".mp3", ".txt").replace(".wav", ".txt")}', mime='text/plain', log_message="Action: Text File Downloaded")
                    with tab2:
                        st.write(result)
                        # Convert result to JSON string
                        json_data = json.dumps(result, indent=4)
                        filename = audio_file[2:]
                        if isinstance(result, dict) and "Stage 1" in result:
                            cleaned_result_stage1 = result["Stage 1"]
                            cleaned_result_stage2 = result.get("Stage 2", [])  # Default to an empty list if Stage 2 is not present
                            overall_result = result.get("Overall Result", "Pass")
                        else:
                            cleaned_result_stage1 = cleaned_result_stage2 = result
                            overall_result = "Pass"

                        # Process Stage 1 results
                        if isinstance(cleaned_result_stage1, list) and all(isinstance(item, dict) for item in cleaned_result_stage1):
                            df_stage1 = pd.json_normalize(cleaned_result_stage1)
                            df_stage1['Stage'] = 'Stage 1'
                        else:
                            df_stage1 = pd.DataFrame(columns=['Stage'])  # Create an empty DataFrame for Stage 1 if no valid results

                        # Process Stage 2 results
                        if isinstance(cleaned_result_stage2, list) and all(isinstance(item, dict) for item in cleaned_result_stage2):
                            df_stage2 = pd.json_normalize(cleaned_result_stage2)
                            df_stage2['Stage'] = 'Stage 2'
                        else:
                            df_stage2 = pd.DataFrame(columns=['Stage'])  # Create an empty DataFrame for Stage 2 if no valid results

                        # Concatenate Stage 1 and Stage 2 results
                        df = pd.concat([df_stage1, df_stage2], ignore_index=True)

                        # Add the Overall Result as a new column (same value for all rows)
                        df['Overall Result'] = overall_result

                        # Add the filename as a new column (same value for all rows)
                        df['Filename'] = filename

                        # Save DataFrame to CSV
                        output = BytesIO()
                        df.to_csv(output, index=False)

                        # Get CSV data
                        csv_data = output.getvalue().decode('utf-8')

                        try:
                            col1, col2 = st.columns([2, 6])
                            with col1:
                                handle_download_json(count=audio_files.index(audio_file) ,data=json_data, file_name=f'{audio_file[2:]}.json', mime='application/json', log_message="Action: JSON File Downloaded")

                            with col2:
                                handle_download_csv(count=audio_files.index(audio_file), data=csv_data, file_name=f'{audio_file[2:]}.csv', mime='text/csv', log_message="Action: CSV File Downloaded")
                        
                        except Exception as e:
                            create_log_entry(f"{e}")
                            st.error(f"Error processing data: {e}")
                    with tab3:
                        zip_buffer = handle_combined_download(
                            data_text=text,
                            data_json=json_data,
                            data_csv=csv_data,
                            file_name_prefix=audio_file[2:]
                        )
                        zip_download(count=audio_files.index(audio_file) ,data=zip_buffer, file_name=f'{audio_file[2:]}.zip', mime="application/zip", log_message="Action: Audited Results Zip File Downloaded")
                        
                current += 1
                create_log_entry(f"Successfully Audited: {audio_file[2:]}")
                df.loc[len(df)] = pd.Series(dtype='float64')
                combined_results.append(df)
                if save_audited_transcript:
                    if save_to_path:
                        try:
                            # Ensure the directory exists
                            os.makedirs(os.path.dirname(save_to_path), exist_ok=True)
                            file_name_without_extension, _ = os.path.splitext(audio_file[2:])
                            full_path = os.path.join(save_to_path, file_name_without_extension + ".csv")
                            # df = pd.json_normalize(csv_data)
                            # Save the DataFrame as a CSV to the specified path
                            df.to_csv(full_path, index=False)
                            print(f"Saved audited results (CSV) to {save_to_path}")
                        except Exception as e:
                            print(f"Failed to save file: {e}")
                    else:
                        print("Save path not specified.")
            if save_audited_transcript:
                if save_to_path:
                    combined_df = pd.concat(combined_results, ignore_index=True)
                    os.makedirs(os.path.dirname(save_to_path), exist_ok=True)
                    full_path = os.path.join(save_to_path, "combined_results.csv")
                    combined_df.to_csv(full_path, index=False)
            # Concatenate all DataFrames
            combined_df = pd.concat(combined_results, ignore_index=True)

            # Create an in-memory CSV using BytesIO
            output = BytesIO()
            combined_df.to_csv(output, index=False)
            output.seek(0)  # Reset buffer position to the start

            # Get the CSV data as a string
            combined_csv_data = output.getvalue().decode('utf-8')
            with st.spinner("Preparing Consolidated Results..."):
                zip_buffer = handle_combined_audit_result_download(
                                data_text=all_text,
                                data_csv=combined_csv_data,
                                file_name_prefix="combined_audit_results"
                            )
                
            combined_audit_result_download(data=zip_buffer, file_name='CombinedAuditResults.zip', mime="application/zip", log_message="Action: Audited Results Zip File Downloaded")  
            directory = "./"
            delete_mp3_files(directory)
        else:
            st.error("Please specify a destination folder to save audited transcript!")


    st.subheader("Event Log")
    log_container = st.container()
    with log_container:
        # Read and display the log content
        log_content = read_log_file()
        log_content = log_content.replace('\n', '<br>')

        # Display the log with custom styling
        html_content = (
            "<div style='height:200px; overflow-y:scroll; background-color:#2b2b2b; color:#f8f8f2; "
            "padding:10px; border-radius:5px; border:1px solid #444;'>"
            "<pre style='font-family: monospace; font-size: 13px; line-height: 1.5em;'>{}</pre>"
            "</div>"
        ).format(log_content)
        
        st.markdown(html_content, unsafe_allow_html=True)
        
    csv_file = 'logfile.csv'
    st.markdown("<br>", unsafe_allow_html=True)
    if os.path.exists(csv_file):
        with open(csv_file, 'rb') as file:
            file_contents = file.read()
            handle_download_log_file(data=file_contents, file_name='log.csv', mime='text/csv', log_message="Action: Event Log Downloaded")

if __name__ == "__main__":
    print(torch.cuda.is_available())  # Should return True if CUDA is set up
    main()