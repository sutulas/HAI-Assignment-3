from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from starlette.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
import os
from dotenv import load_dotenv
import pandas as pd
import altair as alt
import json
import sys
from io import StringIO
import re
from fastapi.responses import JSONResponse  # Import JSONResponse

# python -m uvicorn backend.main:app --reload

# Load environment variables from .env file
load_dotenv()

app = FastAPI()

# Global DataFrame variable to store the uploaded dataset
global_df = pd.DataFrame()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to restrict allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure OpenAI API key
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY")
    # api_key = os.getenv("OPENAI_API_KEY")
)

# Define request and response models
class QueryRequest(BaseModel):
    prompt: str

class QueryResponse(BaseModel):
    response: str

class Spec(BaseModel):
  spec: str

chart = None
chart_description = None
chart_generation_tool = {
  "type": "function",
  "function": {
      "name": "data_visualization_tool",
            "description": "Creates the specifiactions for a vega chart. Call this whenever you have to create a chart, for example: 'mpg v origin'",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "The prompt for the chart",
                    }
                },
                "required": ["prompt"],
                "additionalProperties": False,
            },
  }
}

data_analysis_tool = {
  "type": "function",
  "function": {
      "name": "data_analysis",
            "description": "Creates and runs a python script to analyize data. Call this whenever you have to analyze data, for example: 'average mpg'. Print the output of the code using 'print(...)'",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "The prompt for the analysis",
                    }
                },
                "required": ["prompt"],
                "additionalProperties": False,
            },
  }
}

def generate_chart(query, df):
  prompt = f'''
    Dataset overview (top five rows): {df.head().to_markdown()}

    Given the dataset above, generate a vega-lite specification for the user query, limit width to 300. The data field will be inserted dynamically, so leave it empty: {query}.

  '''
  response = client.beta.chat.completions.parse(
    model="gpt-4o-mini",
    messages=[
      {"role": "user", "content": prompt}
    ],
    response_format=Spec
  )
  return response.choices[0].message.parsed.spec

def generate_code(query, df):
  prompt = f'''
    Dataset overview (top five rows): {df.head().to_markdown()}

    Given the overview of the dataset above, generate python code to answer the user query: {query}.

    Refer the the dataset as 'df' in the code.

    Make sure to print the output of the code using 'print(...)'

    RETURN ONLY THE CODE OR ELSE IT WILL FAIL.

  '''
  response = client.beta.chat.completions.parse(
    model="gpt-4o-mini",
    messages=[
      {"role": "user", "content": prompt}
    ]
  )
  return response.choices[0].message.content

def get_feedback(query, df, spec, type='chart'):
  if type == 'chart':
    builder = "Vega-lite spec"
  else:
    builder = "python code"
  prompt = f'''
    Dataset overview (top five rows): {df.head().to_markdown()}

    User query: {query}.

    Generated {builder}: {spec}

    Please provide feedback on the generated {type}, whether the {builder} is valid in syntax and faithful to the user query.
  '''
  response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
      {"role": "user", "content": prompt}
    ]
  )
  feedback = response.choices[0].message.content
  return feedback

def improve_response(query, df, spec, feedback, type = 'chart'):
  if type == 'chart':
    builder = "Vega-lite spec"
  else:
    builder = "python code"

  prompt = f'''
    Dataset overview (top five rows): {df.head().to_markdown()}

    User query: {query}.

    Generated {builder}: {spec}

    Feedback: {feedback}

    Improve the {builder} with the feedback if only necessary. Otherwise, return the original {type}.

  '''
  if type == 'chart':
    response = client.beta.chat.completions.parse(
      model="gpt-4o-mini",
      messages=[
        {"role": "user", "content": prompt}
      ],
      response_format=Spec
    )
    return response.choices[0].message.parsed.spec
  else:
    prompt += f"RETURN ONLY THE CODE OR ELSE IT WILL FAIL."
    response = client.beta.chat.completions.parse(
      model="gpt-4o-mini",
      messages=[
        {"role": "user", "content": prompt}
      ]
    )
    return response.choices[0].message.content

def data_visualization_tool(prompt):
  reduced_df = global_df.head()
  for attempt in range(2):  # Try twice
    try:
      spec = generate_chart(prompt, reduced_df)
      feedback = get_feedback(prompt, reduced_df, spec)
      final_spec = improve_response(prompt, reduced_df, spec, feedback)
      final_spec_parsed = json.loads(final_spec)
      data_records = global_df.to_dict(orient='records')
      final_spec_parsed['data'] = {'values': data_records}
      # Brief description of the Vega chart
      prompt = f"Provide a short, 2 sentence description of the following vega chart: \n\n {final_spec}"
      response = client.chat.completions.create(
          model="gpt-3.5-turbo",
          messages=[{"role": "user", "content": prompt}]
      )
      response_text = response.choices[0].message.content.strip()

      # Convert the Altair chart to a dictionary (Vega-Lite spec)
      chart = alt.Chart.from_dict(final_spec_parsed)
      chart_json = chart.to_json()  # Convert to JSON format

      # Return the chart JSON to the frontend
      return chart_json, response_text
    except Exception as e:
      # Log the error and retry if it's not the last attempt
      print(f"Graph generation error, trying again...")
      if attempt == 1:  # If this was the last attempt
        return None, "Error: graph failed to load after two attempts, please try again."

def sanitize_input(query: str) -> str:
    """Sanitize input to the python REPL.
    Remove whitespace, backtick & python (if llm mistakes python console as terminal
    """

    # Removes `, whitespace & python from start
    query = re.sub(r"^(\s|`)*(?i:python)?\s*", "", query)
    # Removes whitespace & ` from end
    query = re.sub(r"(\s|`)*$", "", query)
    return query
    
def execute_panda_dataframe_code(code):
    """
    Execute the given python code and return the output. 
    References:
    1. https://github.com/langchain-ai/langchain-experimental/blob/main/libs/experimental/langchain_experimental/utilities/python.py
    2. https://github.com/langchain-ai/langchain-experimental/blob/main/libs/experimental/langchain_experimental/tools/python/tool.py
    """
     # Save the current standard output to restore later
    old_stdout = sys.stdout
    # Redirect standard output to a StringIO object to capture any output generated by the code execution
    sys.stdout = mystdout = StringIO()
    try:
		    # Execute the provided code within the current environment
        cleaned_command = sanitize_input(code)
        exec(cleaned_command, {'df': global_df})
        
        # Restore the original standard output after code execution
        sys.stdout = old_stdout
				
				# Return any captured output from the executed code
        return mystdout.getvalue()
    except Exception as e:
        sys.stdout = old_stdout
        return repr(e)

def data_analysis(prompt):
  df = global_df
  for attempt in range(2):  # Try twice
    try:
      code = generate_code(prompt, df)

      feedback = get_feedback(prompt, df, code, "code")

      final_code = improve_response(prompt, df, code, feedback, "code")
      #print(final_code)
      response = execute_panda_dataframe_code(final_code)
      #print(response)
      return response
    except Exception as e:
      # Log the error and retry if it's not the last attempt
      #print(f"Data analysis error, trying again...")
      if attempt == 1:  # If this was the last attempt
        return None, "Error: data analysis failed after two attempts, please try again."

tools = [chart_generation_tool, data_analysis_tool]
tool_map = {
    "data_visualization_tool": data_visualization_tool,
    "data_analysis": data_analysis
}

def tool_calls(prompt):
  messages = [
    {"role": "system", "content": "You are a helpful assistant. Use the supplied tools to assist the user."},
  ]
  # TODO: send the user message and let the model think about which tool to use if any
  messages.append({"role": "user", "content": prompt})
  response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
    tools=tools,
    tool_choice="auto",
  )
  if response.choices[0].message.tool_calls != None:
    tool_call = response.choices[0].message.tool_calls[0]
    name = tool_call.function.name
    arguments = json.loads(tool_call.function.arguments)

    print('calling tool:', name, ' with arguments:', arguments)# leave this
    function_to_call = tool_map[name]

    # call the function with the arguments
    output = function_to_call(**arguments)
    print('output:', output)
    print(response.choices[0].message.tool_calls[0].id)

    function_call_result_message = {
        "role": "tool",
        "content": json.dumps({
            "arguments": arguments,
            "output": output
        }),
        "tool_call_id": response.choices[0].message.tool_calls[0].id
    }

    messages.append(response.choices[0].message)
    messages.append(function_call_result_message)
    print('messages:', messages)
    response = client.chat.completions.create(
      model="gpt-4o-mini",
      messages=messages,
      tools=tools,
      tool_choice="auto",
    )
    # return the model's observation to the user
    return name, response.choices[0].message.content, output

  # 2. if tool_calls == None
  messages.append(response.choices[0].message)
  return response.choices[0].message.content

def query(question, system_prompt, max_iterations=10):
    global chart
    global chart_description
    # print("dd",pd.read_csv('static/uploads/cars-w-year.csv').head())
    messages = [{"role": "system", "content": system_prompt}]
    messages.append({"role": "user", "content": question})
    i = 0
    while i < max_iterations:
        i += 1
        print("iteration:", i)
        response = client.chat.completions.create(
            model="gpt-4o-mini", temperature=0.0, messages=messages, tools=tools
        )
        # print(response.choices[0].message)
        if response.choices[0].message.content != None:
            print(response.choices[0].message.content)
        # print(response.choices[0].message)

        # if not function call
        if response.choices[0].message.tool_calls == None:
            break

        # if function call
        messages.append(response.choices[0].message)
        for tool_call in response.choices[0].message.tool_calls:
            print("calling:", tool_call.function.name, "with", tool_call.function.arguments)
            # call the function
            arguments = json.loads(tool_call.function.arguments)
            function_to_call = tool_map[tool_call.function.name]
            result = function_to_call(**arguments)
            if tool_call.function.name == "data_visualization_tool":
              chart = result[0]
              chart_description = result[1]
            # create a message containing the result of the function call
            result_content = json.dumps({**arguments, "result": result})
            function_call_result_message = {
                "role": "tool",
                "content": result_content,
                "tool_call_id": tool_call.id,
            }
            #print_blue("action result:", truncate_string(result_content))

            messages.append(function_call_result_message)
        if i == max_iterations and response.choices[0].message.tool_calls != None:
            print("Max iterations reached")
            return "The tool agent could not complete the task in the given time. Please try again."
    print("final response:", response.choices[0].message.content)
    return response.choices[0].message.content


# Endpoint to interact with OpenAI API and generate the chart
@app.options("/query")
async def preflight():
    return JSONResponse(headers={"Access-Control-Allow-Methods": "POST"})

@app.post("/query", response_model=QueryResponse)
async def query_openai(request: QueryRequest):
    global global_df

    if global_df.empty:
        return QueryResponse(response="No dataset uploaded yet.")
    
    # Create a prompt using the dataset
    columns = global_df.columns.tolist()
    prompt = f"Is the following prompt relevant and answerable based on data with these columns {columns}? Any question that mentions the columns is answerable.\n\nRespond with just 'yes' or 'no'.\n\nHere is the prompt: {request.prompt}"

    try:
        # Initial query to check relevance
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        response_text = response.choices[0].message.content.strip()

        if 'yes' in response_text.lower():  # Adjust based on actual check logic
          response = query(request.prompt, "You are a helpful assistant. Use the supplied tools to assist the user.")
          if chart:
            return JSONResponse(content={"chart": json.loads(chart), "response": chart_description})
          else:
            return QueryResponse(response=response)
            # name, response, output = tool_calls(request.prompt)
            # # response = data_analysis(request.prompt, global_df)
            # if name == 'data_visualization_tool':
            #   if output:
            #     return JSONResponse(content={"chart": json.loads(output[0]), "response": output[1]})
            #   else:
            #     return QueryResponse(response="Error: graph failed to load after two attempts, please try again.")
            # else:
            #   return QueryResponse(response=response)
            # reduced_df = global_df.head()
            # chart_json, response_text = data_visualization_tool(request.prompt, reduced_df)
            # if chart_json:  
            #   return JSONResponse(content={"chart": json.loads(chart_json), "response": response_text})
            # else:
            #   return QueryResponse(response="Error: graph failed to load after two attempts, please try again.")

        else:
            return QueryResponse(response=f"The question \"{request.prompt}\" is not relevant to the dataset.")

    except Exception as e:
        return QueryResponse(response=f"Error querying OpenAI: {e}")




# Endpoint to handle file uploads
@app.post("/uploadfile/")
async def upload_file(file: UploadFile = File(...)):
    global global_df  # Access the global DataFrame
    try:
        # Read the uploaded file as a pandas DataFrame
        global_df = pd.read_csv(file.file)

        # Get the title of the first column
        first_column_title = global_df.columns[0]

        # Print "file received" and the first column title
        #print(f"File received. First column title: {first_column_title}")

        # Return a response with a message and the first column title
        return {"message": f"File received, first_column_title: {first_column_title}"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing file: {e}")

# Serve React static files
app.mount("/", StaticFiles(directory="client/build", html=True), name="static")

# Custom 404 handler for React routes
@app.get("/{path_name:path}")
async def serve_react(path_name: str):
    return FileResponse("client/build/index.html")
