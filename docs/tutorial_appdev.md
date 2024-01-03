---
layout: default
title: Try it out!
nav_order: 2
has_children: false
---

# Tutorial: Inferring an AI model from a client application with Caikit
{: .no_toc }

In this tutorial, you will learn how to load an example AI model by using the Caikit runtime and then infer it from a client application by using the provided API.

The example model is [Hugging Face DistilBERT base uncased finetuned SST-2](https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english), which analyzes text for sentiment.

## Prerequisites

- Linux/MacOS x86_64
- [Caikit](https://github.com/caikit/caikit) (v0.9.2)
- [Python](https://www.python.org) (v3.8+)
- [pip](https://pypi.org/project/pip/) (v23.0+)

## What you'll build

- You'll create a Python project that contains the Caikit runtime, an AI model, and a client application.
- The Caikit runtime consists of a data model that representes the attributes of the AI model in code and a module that enables the runtime to load and run the AI model.
- You'll also configure a client application that can use the runtime to querry the AI model for text sentiment analysis on text samples that are supplied by the application.

When you finish, your Python package will have the following directory structure:

```bash
├── text-sentiment/                     # top-level package directory
│   ├── start_runtime.py                # a wrapper to start the Caikit runtime as a gRPC server. The runtime will load the model at startup
│   ├── client.py                       # client which calls the Caikit runtime to perform inference on the model it is serving to perform text sentiment analysis
│   ├── requirements.txt                # specifies library dependencies
│   ├── models/                         # a directory that contains the Caikit metadata of the model and any artifacts required to run the model
│   │   ├── text_sentiment/config.yml   # metadata that defines the Caikit text sentiment model
│   ├── text_sentiment/                 # a directory that defines Caikit module(s) that can include algorithm(s) implementation that can train/run an AI model
│   │   ├── config.yml                  # configuration for the module and model input and output
│   │   ├── __init__.py                 # makes the data_model and runtime_model packages visible
│   │   ├── data_model/                 # a directory that contains the data format of the Caikit module
│   │   │   ├── classification.py       # data class that represents the AI model attributes in code
│   │   │   ├── __init__.py             # makes the data model class visible in the project
│   │   ├── runtime_model/              # a directory that contains the Caikit module of the model
│   │   │   ├── hf_module.py            # a class that bootstraps the AI model in Caikit so it can be served and used (infer/train)
└── └── └── └── __init__.py             # makes the module class visible in the project
```

## Procedure

Complete the following tasks to configure the Caikit runtime and the AI model and test them from a client application:

1. [Create the project](#1-create-the-project)
2. [Create the data model specification](#2-create-the-data-model-specification)
3. [Create the model wrapper](#3-create-the-model-wrapper)
4. [Include the module and package dependencies](#4-include-the-module-and-package-dependencies)
5. [Start the Caikit runtime](#5-start-the-caikit-runtime)
6. [Test the sentiment analysis](#6-test-the-sentiment-analysis)

### 1. Create the project

Run the following commands to create a new project and Python package:

```shell
mkdir -p $HOME/projects/text-sentiment/text_sentiment
cd $HOME/projects/text-sentiment/text_sentiment
```

**Note: The `text_sentiment` package name must use an underscore (`_`), not a dash (`-`) for importing packages.**

### 2. Create the data model specification

The specification defines the data model classes that represent the AI model attributes in code.

1. Run the following commands to create a `data_model` directory to store the data model:

   ```shell
   mkdir data_model
   cd data_model
   ```

    **Note: The Caikit runtime default data model specification package name is `data_model` in the root package directory.**

2. In the `data_model` directory, create a `classification.py` file with the following code:

   ```python
   # Copyright The Caikit Authors
   #
   # Licensed under the Apache License, Version 2.0 (the "License");
   # you may not use this file except in compliance with the License.
   # You may obtain a copy of the License at
   #
   #     http://www.apache.org/licenses/LICENSE-2.0
   #
   # Unless required by applicable law or agreed to in writing, software
   # distributed under the License is distributed on an "AS IS" BASIS,
   # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   # See the License for the specific language governing permissions and
   # limitations under the License.

   # Standard
   from typing import List

   # Local
   from caikit.core import DataObjectBase
   from caikit.core.data_model import dataobject


   @dataobject(package="text_sentiment.data_model")
   class ClassInfo(DataObjectBase):
      """A single classification prediction."""

      class_name: str  # (required) Predicted relevant class name
      confidence: float  # (required) The confidence-like score of this prediction in [0, 1]


   @dataobject(package="text_sentiment.data_model")
   class ClassificationPrediction(DataObjectBase):
      """The result of a classification prediction."""

      classes: List[ClassInfo]


   @dataobject(package="text_sentiment.data_model")
   class TextInput(DataObjectBase):
      """A sample `domain primitive` input type for this library.
      The analog to a `Raw Document` for the `Natural Language Processing` domain."""

      text: str

   ```

3. Make the classes visible in the project by creating an `__init__.py` file with the following content:

   ```python
   # Copyright The Caikit Authors
   #
   # Licensed under the Apache License, Version 2.0 (the "License");
   # you may not use this file except in compliance with the License.
   # You may obtain a copy of the License at
   #
   #     http://www.apache.org/licenses/LICENSE-2.0
   #
   # Unless required by applicable law or agreed to in writing, software
   # distributed under the License is distributed on an "AS IS" BASIS,
   # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   # See the License for the specific language governing permissions and
   # limitations under the License.

   # Local
   from .classification import ClassificationPrediction, ClassInfo, TextInput
   ```

### 3. Create the model wrapper

Create a module class that wraps the example Hugging Face AI model so that Caikit can load and run it.

1. Run the following commands to create a directory for the metadata that identifies the class.

   ```shell
   cd $HOME/projects/text-sentiment
   mkdir -p models/text_sentiment
   cd models/text_sentiment
   ```

    **Note: The Caikit runtime default directory for model metatdata is `models` under the project root directory.**

2. Create a `config.yml` file and add the following metadata to it:

   ```yaml
   # Copyright The Caikit Authors
   #
   # Licensed under the Apache License, Version 2.0 (the "License");
   # you may not use this file except in compliance with the License.
   # You may obtain a copy of the License at
   #
   #     http://www.apache.org/licenses/LICENSE-2.0
   #
   # Unless required by applicable law or agreed to in writing, software
   # distributed under the License is distributed on an "AS IS" BASIS,
   # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   # See the License for the specific language governing permissions and
   # limitations under the License.

   module_id: 8f72161-c0e4-49b0-8fd0-7587b3017a35
   name: HuggingFaceSentimentModule
   version: 0.0.1
   ```

3. Create a directory for the module/wrapper class:

   ```shell
   cd $HOME/projects/text-sentiment/text_sentiment
   mkdir runtime_model
   cd runtime_model
   ```

4. Create an `hf_module.py` file and add the following code to it:

   ```python
   # Copyright The Caikit Authors
   #
   # Licensed under the Apache License, Version 2.0 (the "License");
   # you may not use this file except in compliance with the License.
   # You may obtain a copy of the License at
   #
   #     http://www.apache.org/licenses/LICENSE-2.0
   #
   # Unless required by applicable law or agreed to in writing, software
   # distributed under the License is distributed on an "AS IS" BASIS,
   # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   # See the License for the specific language governing permissions and
   # limitations under the License.

   # Standard
   import os

   # Third Party
   from transformers import pipeline

   # Local
   from caikit.core import ModuleBase, ModuleLoader, ModuleSaver, TaskBase, module, task
   from text_sentiment.data_model.classification import (
      ClassificationPrediction,
      ClassInfo,
      TextInput,
   )


   @task(
      required_parameters={"text_input": TextInput},
      output_type=ClassificationPrediction,
   )
   class HuggingFaceSentimentTask(TaskBase):
      pass


   @module(
      "8f72161-c0e4-49b0-8fd0-7587b3017a35",
      "HuggingFaceSentimentModule",
      "0.0.1",
      HuggingFaceSentimentTask,
   )
   class HuggingFaceSentimentModule(ModuleBase):
      """Class to wrap sentiment analysis pipeline from HuggingFace"""

      def __init__(self, model_path) -> None:
         super().__init__()
         loader = ModuleLoader(model_path)
         config = loader.config
         model = pipeline(model=config.hf_artifact_path, task="sentiment-analysis")
         self.sentiment_pipeline = model

      def run(self, text_input: TextInput) -> ClassificationPrediction:
         """Run HF sentiment analysis
         Args:
               text_input: TextInput
         Returns:
               ClassificationPrediction: predicted classes with their confidence score.
         """
         raw_results = self.sentiment_pipeline([text_input.text])

         class_info = []
         for result in raw_results:
               class_info.append(
                  ClassInfo(class_name=result["label"], confidence=result["score"])
               )
         return ClassificationPrediction(class_info)

      @classmethod
      def bootstrap(cls, model_path="distilbert-base-uncased-finetuned-sst-2-english"):
         """Load a HuggingFace based caikit model
         Args:
               model_path: str
                  Path to HuggingFace model
         Returns:
               HuggingFaceModel
         """
         return cls(model_path)

      def save(self, model_path, **kwargs):
         module_saver = ModuleSaver(
               self,
               model_path=model_path,
         )

         # Extract object to be saved
         with module_saver:
               # Make the directory to save model artifacts
               rel_path, _ = module_saver.add_dir("hf_model")
               save_path = os.path.join(model_path, rel_path)
               self.sentiment_pipeline.save_pretrained(save_path)
               module_saver.update_config({"hf_artifact_path": rel_path})

      # this is how you load the model, if you have a caikit model
      @classmethod
      def load(cls, model_path):
         """Load a HuggingFace based caikit model
         Args:
               model_path: str
                  Path to HuggingFace model
         Returns:
               HuggingFaceModel
         """
         return cls(model_path)

   ```

5. Make the class visible in the project by creating an `__init__.py` file and adding the following to it:

   ```python
   # Copyright The Caikit Authors
   #
   # Licensed under the Apache License, Version 2.0 (the "License");
   # you may not use this file except in compliance with the License.
   # You may obtain a copy of the License at
   #
   #     http://www.apache.org/licenses/LICENSE-2.0
   #
   # Unless required by applicable law or agreed to in writing, software
   # distributed under the License is distributed on an "AS IS" BASIS,
   # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   # See the License for the specific language governing permissions and
   # limitations under the License.
   # # Local
   # Local
   from .hf_module import HuggingFaceSentimentModule
   ```

6. Provide library-specific configuration to be used by the runtime by creating a `config.yml` file.

    Return to the top-level Python package:

   ```shell
   cd $HOME/projects/text-sentiment/text_sentiment/
   ```

    Create a `config.yml` file with the following content:

   ```yaml
   # Copyright The Caikit Authors
   #
   # Licensed under the Apache License, Version 2.0 (the "License");
   # you may not use this file except in compliance with the License.
   # You may obtain a copy of the License at
   #
   #     http://www.apache.org/licenses/LICENSE-2.0
   #
   # Unless required by applicable law or agreed to in writing, software
   # distributed under the License is distributed on an "AS IS" BASIS,
   # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   # See the License for the specific language governing permissions and
   # limitations under the License.

   runtime:
      library: text_sentiment
   ```

7. Configure the library with the configuration file path and make the `data_model` and `runtime_model` packages visible by adding them to an `__init__.py` file.

    Create an `__init__.py` file in the same top-level `text_sentiment` directory as the `config.yml` file and add the following content:

   ```python
   # Copyright The Caikit Authors
   #
   # Licensed under the Apache License, Version 2.0 (the "License");
   # you may not use this file except in compliance with the License.
   # You may obtain a copy of the License at
   #
   #     http://www.apache.org/licenses/LICENSE-2.0
   #
   # Unless required by applicable law or agreed to in writing, software
   # distributed under the License is distributed on an "AS IS" BASIS,
   # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   # See the License for the specific language governing permissions and
   # limitations under the License.

   # Standard
   import os

   # Local
   from . import data_model, runtime_model
   import caikit

   # Give the path to the `config.yml`
   CONFIG_PATH = os.path.realpath(os.path.join(os.path.dirname(__file__), "config.yml"))

   caikit.configure(CONFIG_PATH)

   ```

### 4. Include the module and package dependencies

1. Go to the project root directory:

   ```shell
   cd $HOME/projects/text-sentiment
   ```

2. Create a `requirements.txt` file and add the following dependencies:

   ```text
   caikit

   # Only needed for Hugging Face
   scipy
   torch
   transformers~=4.27.2
   ```

    **Note: Before you go any further, it is advisable to use a [virtual environment(venv)](https://docs.python.org/3/library/venv.html) to avoid conflicts in your environment.**

3. Install the dependencies by running the following command:

```shell
pip install -r requirements.txt
```

### 5. Start the Caikit runtime

The Caikit runtime serves the Hugging Face model so that it can be called for inference. The runtime starts as a [gRPC](https://grpc.io) server, which loads the model on startup. We can then call the model to do sentiment analysis.

1. Create `start_runtime.py` file to start the runtime server.

    You must set the correct path to `import text_sentiment` based on where this `start_runtime.py` file is placed. In the following example, we assume that `start_runtime` file is at the same level as the `text_sentiment` package.

   ```python
   # Copyright The Caikit Authors
   #
   # Licensed under the Apache License, Version 2.0 (the "License");
   # you may not use this file except in compliance with the License.
   # You may obtain a copy of the License at
   #
   #     http://www.apache.org/licenses/LICENSE-2.0
   #
   # Unless required by applicable law or agreed to in writing, software
   # distributed under the License is distributed on an "AS IS" BASIS,
   # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   # See the License for the specific language governing permissions and
   # limitations under the License.

   # Standard
   from os import path
   import sys

   # First Party
   import alog

   sys.path.append(
      path.abspath(path.join(path.dirname(__file__), "../"))
   )  # Here we assume that `start_runtime` file is at the same level of the `text_sentiment` package

   # Local
   import text_sentiment

   alog.configure(default_level="debug")

   # Local
   from caikit.runtime import grpc_server

   grpc_server.main()

   ```

2. In one terminal, start the runtime server by running the following command:

   ```shell
   python3 start_runtime.py
   ```

    You should see output similar to the following example:

   ```vim
   $ python3 start_runtime.py   

   <function register_backend_type at 0x7fce0064b5e0> is still in the BETA phase and subject to change!
   {"channel": "COM-LIB-INIT", "exception": null, "level": "info", "log_code": "<RUN11997772I>", "message": "Loading service module: text_sentiment", "num_indent": 0, "thread_id": 8605140480, "timestamp": "2023-05-02T11:42:52.808812"}
   {"channel": "COM-LIB-INIT", "exception": null, "level": "info", "log_code": "<RUN11997772I>", "message": "Loading service module: caikit.interfaces.common", "num_indent": 0, "thread_id": 8605140480, "timestamp": "2023-05-02T11:42:52.809406"}
   {"channel": "COM-LIB-INIT", "exception": null, "level": "info", "log_code": "<RUN11997772I>", "message": "Loading service module: caikit.interfaces.runtime", "num_indent": 0, "thread_id": 8605140480, "timestamp": "2023-05-02T11:42:52.809565"}
   […]
   {"channel": "MODEL-LOADER", "exception": null, "level": "info", "log_code": "<RUN89711114I>", "message": "Loading model text_sentiment'", "num_indent": 0, "thread_id": 8605140480, "timestamp": "2023-05-02T11:42:52.826657"}
   {"channel": "MDLMNG", "exception": null, "level": "warning", "log_code": "<COR56759744W>", "message": "No backend configured! Trying to configure using default config file.", "num_indent": 0, "thread_id": 8605140480, "timestamp": "2023-05-02T11:42:52.827742"}
   No model was supplied, defaulted to distilbert-base-uncased-finetuned-sst-2-english and revision af0f99b (https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english).
   Using a pipeline without specifying a model name and revision in production is not recommended.
   […]
   {"channel": "COM-LIB-INIT", "exception": null, "level": "info", "log_code": "<RUN11997772I>", "message": "Loading service module: text_sentiment", "num_indent": 0, "thread_id": 8605140480, "timestamp": "2023-05-02T11:42:53.929756"}
   {"channel": "COM-LIB-INIT", "exception": null, "level": "info", "log_code": "<RUN11997772I>", "message": "Loading service module: caikit.interfaces.common", "num_indent": 0, "thread_id": 8605140480, "timestamp": "2023-05-02T11:42:53.929814"}
   {"channel": "COM-LIB-INIT", "exception": null, "level": "info", "log_code": "<RUN11997772I>", "message": "Loading service module: caikit.interfaces.runtime", "num_indent": 0, "thread_id": 8605140480, "timestamp": "2023-05-02T11:42:53.929858"}
   {"channel": "GP-SERVICR-I", "exception": null, "level": "info", "log_code": "<RUN76773778I>", "message": "Validated Caikit Library CDM successfully", "num_indent": 0, "thread_id": 8605140480, "timestamp": "2023-05-02T11:42:53.929942"}
   {"channel": "GP-SERVICR-I", "exception": null, "level": "info", "log_code": "<RUN76884779I>", "message": "Constructed inference service for library: text_sentiment, version: unknown", "num_indent": 0, "thread_id": 8605140480, "timestamp": "2023-05-02T11:42:53.930734"}
   {"channel": "SERVER-WRAPR", "exception": null, "level": "info", "log_code": "<RUN81194024I>", "message": "Intercepting RPC method /caikit.runtime.HfTextsentiment.HfTextsentimentService/HfBlockPredict", "num_indent": 0, "thread_id": 8605140480, "timestamp": "2023-05-02T11:42:53.930786"}
   {"channel": "SERVER-WRAPR", "exception": null, "level": "info", "log_code": "<RUN33333123I>", "message": "Wrapping safe rpc for Predict", "num_indent": 0, "thread_id": 8605140480, "timestamp": "2023-05-02T11:42:53.931424"}
   {"channel": "SERVER-WRAPR", "exception": null, "level": "info", "log_code": "<RUN30032825I>", "message": "Re-routing RPC /caikit.runtime.HfTextsentiment.HfTextsentimentService/HfBlockPredict from <function _ServiceBuilder._GenerateNonImplementedMethod.<locals>.<lambda> at 0x7fce01f660d0> to <function CaikitRuntimeServerWrapper.safe_rpc_wrapper.<locals>.safe_rpc_call at 0x7fce02144670>", "num_indent": 0, "thread_id": 8605140480, "timestamp": "2023-05-02T11:42:53.931479"}
   {"channel": "SERVER-WRAPR", "exception": null, "level": "info", "log_code": "<RUN24924908I>", "message": "Interception of service caikit.runtime.HfTextsentiment.HfTextsentimentService complete", "num_indent": 0, "thread_id": 8605140480, "timestamp": "2023-05-02T11:42:53.931530"}
   […]

   {"channel": "GRPC-SERVR", "exception": null, "level": "info", "log_code": "<RUN10001807I>", "message": "Running in insecure mode", "num_indent": 0, "thread_id": 8605140480, "timestamp": "2023-05-02T11:42:53.936511"}
   {"channel": "GRPC-SERVR", "exception": null, "level": "info", "log_code": "<RUN10001001I>", "message": "Caikit Runtime is serving on port: 8085 with thread pool size: 5", "num_indent": 0, "thread_id": 8605140480, "timestamp": "2023-05-02T11:42:53.938054"}
   ```

### 6. Test the sentiment analysis

The best way to test the model that is loaded is to write some simple Python client code.

1. Create a `client.py` file and add the following code to it:

   ```python
   # Copyright The Caikit Authors
   #
   # Licensed under the Apache License, Version 2.0 (the "License");
   # you may not use this file except in compliance with the License.
   # You may obtain a copy of the License at
   #
   #     http://www.apache.org/licenses/LICENSE-2.0
   #
   # Unless required by applicable law or agreed to in writing, software
   # distributed under the License is distributed on an "AS IS" BASIS,
   # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   # See the License for the specific language governing permissions and
   # limitations under the License.

   # Standard
   import os

   # Third Party
   import grpc

   # Local
   from caikit.runtime.service_factory import ServicePackageFactory
   from text_sentiment.data_model import TextInput

   inference_service = ServicePackageFactory().get_service_package(
      ServicePackageFactory.ServiceType.INFERENCE,
   )

   port = 8085

   # Setup the client
   channel = grpc.insecure_channel(f"localhost:{port}")
   client_stub = inference_service.stub_class(channel)

   # Run inference for two sample prompts
   for text in ["I am not feeling well today!", "Today is a nice sunny day"]:
      input_text_proto = TextInput(text=text).to_proto()
      request = inference_service.messages.HuggingFaceSentimentTaskRequest(
         text_input=input_text_proto
      )
      response = client_stub.HuggingFaceSentimentTaskPredict(
         request, metadata=[("mm-model-id", "text_sentiment")]
      )
      print("Text:", text)
      print("RESPONSE:", response)

   ```

2. Open a new terminal and run the client code:

   ```shell
   python3 client.py
   ```

    The client code calls the model and queries it for sentiment analysis on a 2 different pieces of text, `I am not feeling well today!` and `Today is a nice sunny day`.

    Look for output similar to the following example, which provides sentiment classification and a confidence score for each text sample:

   ```command
   $ python3 client.py

   <function register_backend_type at 0x7fe930bdbdc0> is still in the BETA phase and subject to change!
   Text: I am not feeling well today!
   RESPONSE: classes {
   class_name: "NEGATIVE"
   confidence: 0.99977594614028931
   }

   Text: Today is a nice sunny day
   RESPONSE: classes {
   class_name: "POSITIVE"
   confidence: 0.999869704246521
   }   
   ```

## Results

You configured the Caikit runtime to load and run a Hugging Face text sentiment analysis model. You then ran a client application that calls the Caikit API to query the Hugging Face model for sentiment analysis on text strings. The model response included the sentiment analysis and a confidence score for each sample.
