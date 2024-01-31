---
title: Architecture Layers
layout: default
nav_order: 6
---

# Architecture Layers

The `caikit` project is itself a stack of complimentary building blocks that layer on top of each other to provide rising levels of abstraction to the various functions in the space of AI problems. The layers roughly break down as follows:

1. [**Useful Tools**](#1-useful-tools): At the bottom, `caikit` holds a collection of tools that are not AI-specific. These tools are useful for AI, but can be imported and used in just about any problem space where they fit. These tools handle low-level functions like configuration, logging, concurrency, and more.
    * DEPENDENCY: `pip install caikit` (no extras)

2. [**AI Data and Model Abstractions**](#2-ai-data-and-model-abstractions): Building on the **Useful Tools**, `caikit` provides a set of abstractions that help to frame AI models based on how a user would consume them (as opposed to how a Data Scientist would author them). Like the **Useful Tools**, these abstractions can be imported and used in any project where there is a need for a pluggable set of runnable objects which can be serialized and configured.
    * DEPENDENCY: `pip install caikit` (no extras)

3. [**AI Runtime**](#3-ai-runtime): One of the most common needs in any AI application is a server that can provide API access to performing AI tasks in a repeatable way. These tasks are generally `train` (any form of model creation/customization that results in usable artifacts), and `run` (any form of model inference which takes user input and produces model outputs).
    * DEPENDENCY: `pip install caikit[runtime-XYZ]` (e.g. `runtime-grpc`)

4. [**AI Domain Interfaces**](#4-ai-domain-interfaces): The space of AI is large, and the number of different `domains`, `tasks`, and `data objects` that can be created can be extremely daunting. Many AI platforms attempt to solve this by leaving the interfaces as opaque blobs that only need to be parsable by the model code. This, however, tightly couples the client-side usage of the models to the model implementation which can cause brittleness and migration challenges over time. The `caikit.interfaces` data model attempts to formalize the `domains`, `tasks`, and `data objects` that are most commonly implemented by AI models so that clients can write their code against these interfaces without binding themselves to specific model implementations.
    * DEPENDENCY: `pip install caikit[interfaces-XYZ]` (e.g. `interfaces-vision`)

5. [**AI Domain Libraries**](#5-ai-domain-libraries): Outside of the core `caikit` library, the `caikit` community supports an ever-growing set of domain libraries which provide concrete implementations of the tasks defined in `caikit.interfaces`. These libraries provide tested model implementations that can be used out-of-the box (subject to each project's maturity level). They also encode some of the more advanced usecases for optimized runtimes (e.g. [text-generation using a remote TGIS backend](https://github.com/caikit/caikit-nlp/blob/main/caikit_nlp/modules/text_generation/text_generation_tgis.py))
    * DEPENDENCY: `pip install caikit-XYZ` (e.g. `caikit-nlp`)

6. [**Prebuilt Runtime Images**](#6-prebuilt-runtime-images): The most common way to encapsulate an instance of an **AI Runtime** is in a [OCI container image](https://opencontainers.org/) (e.g. [docker](https://www.docker.com/), [podman](https://podman.io/), [rancher](https://www.rancher.com/)). Such images can be run using container runtimes such as [kubernetes](https://kubernetes.io/). The `caikit` community provides prebuilt images through our work with Red Hat OpenShift AI which can be directly used and run in a user's application. These images are built using a single **AI Domain Library** and are each capable of running the collection of modules defined there.
    * DEPENDENCY: `docker pull caikit-XYZ-runtime`

7. [**Kubernetes Runtime Stack**](#7-kubernetes-runtime-stack): When building a [kubernetes](https://kubernetes.io/) application that manages AI tasks, the `caikit-runtime-stack` provides a convenient [kubernetes operator](https://kubernetes.io/docs/concepts/extend-kubernetes/operator/) which can deploy a collection of **Prebuilt Runtime Images** into a cohesive ML Ops stack with configurable security, scalability, and data transfer options.
    * **NOTE**: We are still in the process of releasing the `caikit-runtime-stack-operator` as open source. Stay tuned!

## 1. Useful Tools

The core of the `caikit` building block tools live in [caikit.config](https://github.com/caikit/caikit/tree/main/caikit/config) and [caikit.core](https://github.com/caikit/caikit/tree/main/caikit/core). This library of tools encapsulates the low-level glue that holds all of the higher-order capabilities together.

### Configuration

Following [standard 12-factor app](https://12factor.net/config) practices, `caikit` provides a robust config mechanism using [alchemy-config (`aconfig`)](https://pypi.org/project/alchemy-config/) that allows config defaults to be versioned in a `yaml` file or python `dict` with override values coming from environment variables and/or override files.

**config_example.py**

```py
import caikit.config
my_config_defaults = {"foo": {"bar": 1, "baz": "two", "biz": True}}
caikit.configure(config_dict=my_config_defaults)
print(caikit.get_config().foo)
```

```sh
# Run with defaults
python config_example.py
# >>> {'bar': 1, 'baz': 'two', 'biz': True}

# Override from the environment
FOO_BIZ=false python config_example.py
# >>> {'bar': 1, 'baz': 'two', 'biz': False}

# Override from a yaml config file
echo -e "foo:\n baz: three\n" > override.yaml
CONFIG_FILES=override.yaml python config_example.py
# >>> {'baz': 'three', 'bar': 1, 'biz': True}
```

### Factories

One of the main tools behind `caikit`'s plugin architecture is the [Factory Pattern](https://en.wikipedia.org/wiki/Factory_method_pattern). This is implemented in `caikit` in [caikit.core.toolkit.factory](https://github.com/caikit/caikit/blob/main/caikit/core/toolkit/factory.py). These base implementations can be useful for building additional pluggable abstractions in user projects.

**factory_example.py**

```py
from caikit.core.toolkit.factory import ImportableFactory, FactoryConstructible
import abc
import aconfig
import caikit
import os

###########
## Bases ##
###########

class DatastoreClient(abc.ABC):
    @abc.abstractmethod
    def put_data(self, key: str, data: str):
        """Abstract put method"""


class DatastoreConnectorBase(FactoryConstructible):
    @abc.abstractmethod
    def get_client(self) -> DatastoreClient:
        """All connectors must be able to get a client handle"""

DATASTORE_FACTORY = ImportableFactory("datastore")

###########
## My DB ##
###########

class MyDBClient(DatastoreClient):

    def __init__(self, hostname: str, port: int):
        # STUB
        self._client = {"hostname": hostname, "port": port}

    def put_data(self, key: str, data: str):
        # STUB
        self._client.write(key, data)


class MyDBConnector(DatastoreConnectorBase):

    name = "MYDB"

    def __init__(self, config: aconfig.Config, instance_name: str):
        self.instance_name = instance_name
        self._client = MyDBClient(config.hostname, config.port)

    def get_client(self) -> MyDBClient:
        return self._client


DATASTORE_FACTORY.register(MyDBConnector)

#############
## Your DB ##
#############

class YourDBClient(DatastoreClient):

    def __init__(self, root_path: str):
        self._root = root_path
        if not os.path.isdir(self._root):
            os.makedirs(self._root)

    def put_data(self, key: str, data: str):
        with open(os.path.join(self._root, key), "w") as handle:
            handle.write(data)


class YourDBConnector(DatastoreConnectorBase):

    name = "YOURDB"

    def __init__(self, config: aconfig.Config, instance_name: str):
        self.instance_name = instance_name
        self._client = YourDBClient(config.root)

    def get_client(self) -> MyDBClient:
        return self._client


DATASTORE_FACTORY.register(YourDBConnector)

##########
## Main ##
##########

if __name__ == "__main__":
    config_defaults = {
        "datastore": {
            "type": "YOURDB",
            "config": {
                "root": ".yourdb"
            }
        }
    }
    caikit.configure(config_dict=config_defaults)
    inst = DATASTORE_FACTORY.construct(caikit.get_config().datastore)
    print(inst.instance_name)
```

```sh
# Default to using YOURDB
python factory_example.py
# >>> YOURDB

# Override to use MYDB
DATASTORE_TYPE=MYDB python factory_example.py
# >>> MYDB
```

### Concurrency Management

AI workloads are often long running, especially when training or tuning a model. Either due to user cancellation or resource shedding, production AI applications need to be able to run work in isolation and cancel that work when needed to recoup resources. The [caikit.core.toolkit.concurrency](https://github.com/caikit/caikit/tree/main/caikit/core/toolkit/concurrency) module has a number of useful tools for managing work in isolation and handling cancellation.

**concurrency_example.py**

```py
from caikit.core.toolkit.concurrency.destroyable_process import DestroyableProcess
import time


def work(message: str = "Hello World", max_duration: int = 20):
    start_time = time.time()
    while time.time() < start_time + max_duration:
        print(f"Message: {message}")
        time.sleep(1)


proc = DestroyableProcess("fork", work)

proc.start()
time.sleep(3)
proc.destroy()
proc.join()
print(f"Destroyed? {proc.destroyed}")
print(f"Canceled? {proc.canceled}")
print(f"Ran? {proc.ran}")
```

```sh
python concurrency_example.py
# >>> Message: Hello World
# >>> Message: Hello World
# >>> Message: Hello World
# >>> Message: Hello World
# >>> Destroyed? True
# >>> Canceled? True
# >>> Ran? True
```

### Logging and Error Reporting

All AI libraries and applications need a solution for application logging and error reporting. In `caikit`, we use [alchemy-logging (`alog`)](https://pypi.org/project/alchemy-logging/) on top of the base [python `logging` package](https://docs.python.org/3/library/logging.html) and implement the [caikit.core.exceptions.error_handler](https://github.com/caikit/caikit/blob/main/caikit/core/exceptions/error_handler.py) module to wrap up common error handling patterns. This combination provides straightforward logging configuration following the [Configuration](#configuration) conventions as well as standardized error condition checks and reporting.

**log_error_example.py**

```py
from caikit.core.exceptions import error_handler
import alog
import caikit

log = alog.use_channel("DEMO")
error = error_handler.get(log)


@alog.logged_function(log.debug)
def doit(data: str, param: int) -> str:
    error.type_check("<DMO00255362E>", str, data=data)
    error.type_check("<DMO58840196E>", int, param=param)
    error.value_check(
        "<DMO62491029E>", param > 0, "Invalid param value (%d). Must be > 0", param
    )
    return data * param


custom_config = {"data": "Hello world", "param": 1}
caikit.configure(config_dict=custom_config)
cfg = caikit.get_config()
alog.configure(
    default_level=cfg.log.level,
    filters=cfg.log.filters,
    formatter=cfg.log.formatter,
    thread_id=cfg.log.thread_id,
)
print(doit(cfg.data, cfg.param))
```

```sh
# Valid
python log_error_example.py
# >>> Hello world

# Valid with log config
LOG_LEVEL=debug LOG_FORMATTER=pretty python log_error_example.py
# >>> 2023-12-11T21:39:29.044954 [DEMO :DBUG:7983813376] BEGIN: doit()
# >>> 2023-12-11T21:39:29.045038 [DEMO :DBUG:7983813376] END: doit()
# >>> Hello world

# Invalid type
DATA=123 python log_error_example.py
# >>> {"channel": "DEMO", "exception": null, "level": "error", "log_code": "<DMO00255362E>", "message": "exception raised: TypeError(\"type check failed: variable `data` has type `int` (fully qualified name `builtins.int`) not in `('str',)`\")", "num_indent": 0, "thread_id": 7983813376, "timestamp": "2023-12-11T21:39:53.383348"}
# >>> ...
# >>> TypeError: type check failed: variable `data` has type `int` (fully qualified name `builtins.int`) not in `('str',)`

# Invalid value
PARAM=0 python log_error_example.py
# >>> {"channel": "DEMO", "exception": null, "level": "error", "log_code": "<DMO62491029E>", "message": "exception raised: ValueError('value check failed: Invalid param value (%d). Must be > 0')", "num_indent": 0, "thread_id": 7983813376, "timestamp": "2023-12-11T21:40:59.665439"}
# >>> ...
# >>> ValueError: value check failed: Invalid param value (%d). Must be > 0
```

## 2. AI Data and Model Abstractions

Above the [**Useful Tools**](#1-useful-tools), `caikit` provides a collection of abstractions and utilities that provide the building blocks for AI workloads. These are all live in `caikit.core` and each have their own sub module.

### Data Modeling

One of the most important aspects of defining AI workloads is defining the shape of the data needed to perform the work. This can be as simple as `str -> str`, but can quickly grow in complexity to include compositional data model objects (e.g. [TokenClassificationResults](https://github.com/caikit/caikit/blob/main/caikit/interfaces/nlp/data_model/classification.py#L104C7-L104C33)).

To support data object definition and usage, `caikit` provides the [@dataobject](https://github.com/caikit/caikit/blob/main/caikit/core/data_model/dataobject.py) decorator. This utility adapts the [python @dataclass decorator](https://docs.python.org/3/library/dataclasses.html#dataclasses.dataclass) and [Enum class](https://docs.python.org/3/library/enum.html) and binds them to [protobuf](https://protobuf.dev/) definitions for serialization and transport.

**dataobject_exmaple.py**

```py
from caikit.core import DataObjectBase, dataobject
from enum import Enum
from typing import List, Iterable, Tuple, Union
import struct


@dataobject
class DataTypes(Enum):
    FLOAT = 1
    INT = 2


@dataobject
class DataBuffer(DataObjectBase):
    data: bytes
    data_type: DataTypes

    @classmethod
    def from_iterable(
        cls, vals: Union[Iterable[int], Iterable[float]],
    ) -> "DataBuffer":
        data_type = (
            DataTypes.INT
            if vals and isinstance(vals[0], int)
            else DataTypes.FLOAT
        )
        data = struct.pack("!I" + "d" * len(vals), len(vals), *vals)
        return cls(data, data_type)

    def to_type(self) -> Union[Tuple[float], Tuple[int]]:
        unpacked = struct.unpack_from(
            "!" + "d" * struct.unpack_from("!I", self.data)[0], self.data, 4
        )
        if self.data_type == DataTypes.INT:
            return tuple(int(element) for element in unpacked)
        return unpacked


@dataobject
class Embeddings(DataObjectBase):
    embeddings: List[DataBuffer]

    def to_types(self) -> List[Union[Tuple[float], Tuple[int]]]:
        return [element.to_type() for element in self.embeddings]


embedding_set = Embeddings([
    DataBuffer.from_iterable([1, 2, 3]),
    DataBuffer.from_iterable([4, 5, 6]),
    DataBuffer.from_iterable([0.7, 0.8, 0.9]),
])

print("## To Types")
print(embedding_set.to_types())
print()

print("## To Json")
print(embedding_set.to_json())
print()

print("## To Protobuf")
print(embedding_set.to_proto())
print()
```

```sh
python dataobject_example.py
# >>> ## To Types
# >>> [(1, 2, 3), (4, 5, 6), (0.7, 0.8, 0.9)]
# >>>
# >>> ## To Json
# >>> {"embeddings": [{"data": "AAAAAz/wAAAAAAAAQAAAAAAAAABACAAAAAAAAA==", "data_type": "INT"}, {"data": "AAAAA0AQAAAAAAAAQBQAAAAAAABAGAAAAAAAAA==", "data_type": "INT"}, {"data": "AAAAAz/mZmZmZmZmP+mZmZmZmZo/7MzMzMzMzQ==", "data_type": "FLOAT"}]}
# >>>
# >>> ## To Protobuf
# >>> embeddings {
# >>>   data: "\000\000\000\003?\360\000\000\000\000\000\000@\000\000\000\000\000\000\000@\010\000\000\000\000\000\000"
# >>>   data_type: INT
# >>> }
# >>> embeddings {
# >>>   data: "\000\000\000\003@\020\000\000\000\000\000\000@\024\000\000\000\000\000\000@\030\000\000\000\000\000\000"
# >>>   data_type: INT
# >>> }
# >>> embeddings {
# >>>   data: "\000\000\000\003?\346ffffff?\351\231\231\231\231\231\232?\354\314\314\314\314\314\315"
# >>>   data_type: FLOAT
# >>> }
```

### Data Streaming

For many AI workloads, especially `training`, data volumes quickly escape the bounds of in-memory iterables. There are numerous solutions for managing iterables of data backed by disk or even remote services. In `caikit`, these are all managed via the [DataStream](https://github.com/caikit/caikit/blob/main/caikit/core/data_model/streams/data_stream.py) utility. This class provides consistent iteration semantics wrapping arbitrary [python generators](https://wiki.python.org/moin/Generators).

**datastream_example.py**

```py
from caikit.core.data_model import DataStream
import os

in_memory_stream = DataStream.from_iterable((1, 2, 3, 4))
print(list(in_memory_stream))
print(list(in_memory_stream))

io_bound_stream = DataStream(lambda: iter(os.listdir(".")))
print(list(io_bound_stream))
print(list(io_bound_stream))
```

```sh
python datastream_example.py
# >>> [1, 2, 3, 4]
# >>> [1, 2, 3, 4]
# >>> ['_includes', 'LICENSE', 'code-of-conduct.md', 'CODEOWNERS', 'config_example.py', 'docs', '_layouts', 'README.md', '.gitignore', 'index.md', '_config.yml', '.github', 'Gemfile', 'Gemfile.lock', '.git', 'assets']
# >>> ['_includes', 'LICENSE', 'code-of-conduct.md', 'CODEOWNERS', 'config_example.py', 'docs', '_layouts', 'README.md', '.gitignore', 'index.md', '_config.yml', '.github', 'Gemfile', 'Gemfile.lock', '.git', 'assets']
```

### Modules

The basic unit of work for AI is the `model`. An AI `model` is, at its highest level, an attempt to replicate some process which we (people) would deem to require intelligence. For the vast majority of models, the behavior of the model is determined by two things:

1. The **algorithm**: This defines _how_ the model attempts to replicate the process
2. The **data**: This defines _what_ the model is attempting to replicate (observations of the process in the real world)

Combining these two ingredients into a concrete asset is typically framed as the `training` work. Using the resulting combined asset to replicate the process is the `inference` work.

One key observation is that when an `inference` job executes, it _must_ use the same **algorithm** that was used to create the `model` artifacts in the first place, otherwise there can be no guarantee that the model will accurately reproduce the learned approximation to the process that resulted from `training`. In software, this often means controlling as many variables as possible between `training` and `inference`. These variables can include things like preprocessing steps, software package versions, hardware driver versions, etc.

In `caikit`, the [module](https://github.com/caikit/caikit/blob/main/caikit/core/modules/base.py) abstraction provides the base for implementing the **algorithm** for both `training` and `inference` in a single location so that these variables can be explicitly controlled for. The key functions that a `module` expects are `train`, `save`, `load`, and `run`. Depending on the usecase, each of these may be optional, but canonically a `module` which defines all of these functions provides a fully encapsulated and repeatable model template.

**module_example.py**

```py
from caikit.core import ModuleBase, ModuleConfig, ModuleLoader, ModuleSaver, module
from caikit.core.data_model import DataStream
from typing import List, Tuple, Union
import argparse
import re
import tempfile


@module(
    "0f148161-90fc-4275-b6b5-6cbdf9826af6",
    "NameFinder",
    "1.0.0",
)
class NameFinder(ModuleBase):

    def __init__(self, name_list: List[str]):
        self._name_list = set(name_list)

    def run(self, text: str) -> List[str]:
        return [
            name for name in self._name_list
            if name in self._preprocess(text)
        ]

    @classmethod
    def train(
        cls,
        training_data: DataStream[str],
        delimiters = (" ", ".", ",", ":", ";", "'", "\""),
        ngram_size: int = 2,
    ) -> "NameFinder":
        names = []
        expr = re.compile("[{}]".format("".join(delimiters)))
        for sample in training_data:
            sample = cls._preprocess(sample)
            token_boundaries = cls._get_token_boundaries(sample, expr)
            for ng_len in range(1, ngram_size + 1):
                for idx in range(len(token_boundaries)):
                    if idx + ng_len > len(token_boundaries):
                        break
                    tokens = token_boundaries[idx:idx + ng_len]
                    names.append(sample[tokens[0][0]: tokens[-1][1]])
        return cls(names)

    def save(self, model_path: str):
        with ModuleSaver(self, model_path=model_path) as module_saver:
            module_saver.update_config({
                "name_list": self._name_list,
            })

    @classmethod
    def load(cls, model_path: Union[str, ModuleConfig]) -> "NameFinder":
        loader = ModuleLoader(model_path)
        return cls(loader.config["name_list"])

    ## Impl ##

    @staticmethod
    def _preprocess(text: str) -> str:
        return text.lower()

    @staticmethod
    def _get_token_boundaries(
        sample: str, expr: re.Pattern,
    ) -> List[Tuple[int, int]]:
        token_boundaries = []
        start = 0
        for match in expr.finditer(sample):
            match_offsets = match.span()
            token_boundaries.append((start, match_offsets[0]))
            start = match_offsets[1]
        if start < len(sample):
            token_boundaries.append((start, len(sample)))
        return token_boundaries


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Caikit module example")
    parser.add_argument(
        "--training-sample", "-s", nargs="+", required=True, help="Names to train with",
    )
    parser.add_argument(
        "--test-sample", "-t", nargs="+", required=True, help="Sample text to run inference with",
    )
    parser.add_argument(
        "--ngram_size", "-n", type=int, default=None, help="N-Gram length for names",
    )
    parser.add_argument(
        "--delimiters", "-d", nargs="*", default=None, help="Tokenization delimiters for training samples",
    )
    args = parser.parse_args()

    # Train the model
    train_kwargs = {"training_data": DataStream.from_iterable(args.training_sample)}
    if args.delimiters is not None:
        train_kwargs["delimiters"] = args.delimiters
    if args.ngram_size is not None:
        train_kwargs["ngram_size"] = args.ngram_size
    trained_model = NameFinder.train(**train_kwargs)

    # Run the samples through the trained model
    print("## Trained Model ##")
    for sample in args.test_sample:
        res = trained_model.run(sample)
        print(f"Results for [{sample}]: {res}")
    print()

    # Save and reload the model
    with tempfile.TemporaryDirectory() as workdir:
        trained_model.save(workdir)
        loaded_model = NameFinder.load(workdir)

    # Rerun with the loaded model
    print("## Loaded Model ##")
    for sample in args.test_sample:
        res = trained_model.run(sample)
        print(f"Results for [{sample}]: {res}")
    print()
```

```sh
python module_example.py \
    -s "Gabe Goodhart" "Gabriel Goodhart" "Gabriel Lincoln Goodhart" \
    -t "Hi, my name is Gabriel Lincoln Goodhart, but most folks all me Gabe"
# >>> ## Trained Model ##
# >>> Results for [Hi, my name is Gabriel Lincoln Goodhart, but most folks all me Gabe]: ['lincoln goodhart', 'goodhart', 'gabriel lincoln', 'gabriel', 'gabe', 'lincoln']
# >>>
# >>> ## Loaded Model ##
# >>> Results for [Hi, my name is Gabriel Lincoln Goodhart, but most folks all me Gabe]: ['lincoln goodhart', 'goodhart', 'gabriel lincoln', 'gabriel', 'gabe', 'lincoln']
```

### Tasks

In AI workloads, a `task` typically refers to an abstract problem or process that can be solved (modeled) by one or more AI algorithms. In `caikit`, we use the `caikit.core.task` module to define specific `tasks` based on the required input and output data types. These act as an abstract function signature for `Modules` that implement a model algorithm to solve the given task.

Some tasks with sequential intput/output types can have multiple flavors of inference signature to account for intput and output streaming (e.g. `text generation` which can produce its output as a single string, or a stream of tokens). The `@task` decorator in `caikit` allows a single `task` to bind these multiple inference flavors together into a single logical task abstraction. Implementations of the `task` may choose which of the signatures to implement.

**task_example.py**

```py
from caikit.core import ModuleBase, TaskBase, module, task
from caikit.core.data_model import DataObjectBase, DataStream, dataobject
from caikit.core.exceptions import error_handler
from typing import Dict, Iterable
import alog

log = alog.use_channel("DEMO")
error = error_handler.get(log)

@dataobject
class Document(DataObjectBase):
    data: str


@dataobject
class TranslationRequestChunk(DataObjectBase):
    text: str
    target_language: str


@task(
    unary_parameters={"text": str, "target_language": str},
    unary_output_type=Document,
    streaming_parameters={"chunks": Iterable[TranslationRequestChunk]},
    streaming_output_type=Iterable[str],
)
class LanguageTranslationTask(TaskBase):
    """This task translates from one language to another target language"""


@module(
    "59c226a0-aabe-478d-841a-3bd6a030a897",
    "Sample LT Module",
    "1.0.0",
    task=LanguageTranslationTask,
)
class SampleLT(ModuleBase):

    def __init__(self, word_map: Dict[str, Dict[str, str]]):
        self._word_map = word_map

    @LanguageTranslationTask.taskmethod(input_streaming=True, output_streaming=True)
    def run_bidi_stream(
        self, chunks: DataStream[TranslationRequestChunk],
    ) -> DataStream[str]:

        def translation_generator():
            for chunk in chunks:
                text = chunk.text
                target_language = chunk.target_language
                error.value_check(
                    "<DMO15854362E>",
                    target_language in self._word_map,
                    "Unsupported target language: %s",
                    target_language,
                )
                target_word_map = self._word_map[target_language]
                for word in text.split():
                    yield target_word_map.get(word, f"UNK[{word}]")

        return DataStream(translation_generator)

    @LanguageTranslationTask.taskmethod()
    def run(self, text: str, target_language: str) -> Document:
        return Document(
            " ".join(
                self.run_bidi_stream([TranslationRequestChunk(text, target_language)])
            )
        )


if __name__ == "__main__":
    model = SampleLT({"foo": {"foo": "bar", "baz": "bat"}})
    print(model.run("foo is a bar", "foo"))

    input_stream = DataStream.from_iterable([
        TranslationRequestChunk("foo is", "foo"),
        TranslationRequestChunk("a bar", "foo"),
    ])
    for result in model.run_bidi_stream(input_stream):
        print(result)
```

```sh
python task_example.py
# >>> {
# >>>   "data": "bar UNK[is] UNK[a] UNK[bar]"
# >>> }
# >>> bar
# >>> UNK[is]
# >>> UNK[a]
```

### Model Management

At runtime, there are a set of common usage patterns for `modules` and `models` that are defined in `caikit.core.model_management` to manage the in-process state of a given `model` instance. In the abstract, those patterns are:

* `trainer` ([ModelTrainerBase](https://github.com/caikit/caikit/blob/main/caikit/core/model_management/model_trainer_base.py)): A `trainer` is responsible for creating a usable `model` instance from raw materials (pretrained base models, training data, parameters). The local implementation simply wraps the `module`'s `train` function. Alternate implementations may delegate the training workload elsewhere beyond the local process/container/machine.

* `finder` ([ModelFinderBase](https://github.com/caikit/caikit/blob/main/caikit/core/model_management/model_finder_base.py)): A `finder` is responsible for locating the artifacts for a concrete `model` that can be loaded into memory for inference. The local implementation locates file artifacts on disk using the standard [ModuleConfig](https://github.com/caikit/caikit/blob/main/caikit/core/modules/config.py) formatted `config.yml` file. Alternate implementations may locate a model running remodely, or parse an alternate metadata format (e.g. read a `transformers` model's [config.json](https://github.com/huggingface/transformers/blob/415e9a0980b00ef230d850bff7ecf0021c52640d/src/transformers/utils/__init__.py#L217)). The output of a `finder` is an in-memory `ModuleConfig` object.

* `initializer` ([ModelInitializerBase](https://github.com/caikit/caikit/blob/main/caikit/core/model_management/model_initializer_base.py)): An `initializer` is responsible for taking a `ModuleConfig` and creating a running `module` instance that can be used for inference. The local implementation simply uses the `load` funciton from the corresponding `module` class to load the model into memory. Alternate implementations may create proxy objects to an instance of a model running elsewhere (e.g. generic client handle to a `caikit.runtime` server running elsewhere with the model pre-loaded).

Each of these abstractions can be used via the top-level functions [caikit.train](https://github.com/caikit/caikit/blob/main/caikit/core/model_manager.py#L94) and [caikit.load](https://github.com/caikit/caikit/blob/main/caikit/core/model_manager.py#L204). These functions are defined on the singleton `ModelManager` instance. When a model-management component is not explicitly passed to `caikit.train`/`caikit.load`, the `"default"` is used based on the configuration in `model_management.[trainers|finders|initializers].default` ([here](https://github.com/caikit/caikit/blob/main/caikit/config/config.yml#L27)).

### Augmentors

One of the most common operations when trainig or tuning an AI model is to perform [data augmentation](https://en.wikipedia.org/wiki/Data_augmentation) on the training data to improve how well it represents the statistics of the behavior being modeled. In `caikit`, this is managed using the [caikit.core.augmentors](https://github.com/caikit/caikit/blob/main/caikit/core/augmentors/__init__.py) abstractions. The core interface of an `Augmentor` is a basic data filter where the output type must match the input type.

```py
import random
from caikit.core.augmentors import AugmentorBase
from caikit.core.data_model import DataStream


class DoublingAugmentor(AugmentorBase):

    augmentor_type = int

    def _augment(self, val: int) -> int:
        return val * 2


class RandomNoiseAugmentor(AugmentorBase):

    augmentor_type = float

    def __init__(self, random_seed: int = 42, range_size: float = 1.0):
        self._range_size = range_size
        super().__init__(random_seed, False)

    def _augment(self, val: float) -> float:
        delta = (random.random() - 0.5) * self._range_size
        return val + delta


def int_to_float(val: int) -> float:
    return float(val)


# Create a data stream that has been augmented with the doubled sequence and the
# randomly permuted sequence
stream = DataStream.from_iterable(range(10)).augment(
    DoublingAugmentor(42, False), 1, post_augment_func=int_to_float,
).augment(
    RandomNoiseAugmentor(), 1,
)
print(list(stream))
```

## 3. AI Runtime

For production AI applications, the most common pattern of usage is to provide model functionality [As A Service](https://en.wikipedia.org/wiki/As_a_service). The `caikit.runtime` module provides a server with two possible interfaces that can serve both `training` and `inference` requests for `caikit` models.

The simplest way to run `caikit.runtime` is `python -m caikit.runtime`. This will launch the desired server interface heads based on the config values `runtime.grpc.enabled` and `runtime.http.enabled`.

### Service Introspection

Since `caikit` is designed to manage AI [tasks](#tasks) in the abstract, `caikit.runtime` does not encode any _explicit_ tasks in its APIs. Instead, it inspects the selection of `module` implementations available and creates `training` and `inference` APIs dynamically at boot.

The set of `modules` available is controlled by setting the [runtime.libarary](https://github.com/caikit/caikit/blob/main/caikit/config/config.yml#L94) configuration. This will cause the referenced library to be imported at boot time and all `@module` decorators will auto-register the corresponding module class.

The `training` service will create an endpoint for each available `module`'s `train` function. The typed arguments for the `train` function will be inspected to form a `DataObject` with the corresponding key names and value types.

The `inference` service will create an endpoint for each [task](#task) that has one or more `module` implementations available. Inference requests will require all of the `task`'s input parameters to be given and aggregate additional arguments from all availble implementations into a task inference request.

The set of interfaces for a given `caikit.runtime` with a given `runtime.library` can be dumped using `python -m caikit.runtime.dump_services`. The output files can then be used to create client-side code that will make requests against the running server.

### gRPC Server

The `caikit.runtime.grpc_server` module runs a [grpc](https://grpc.io/) server with `RPCs` for each endpoint in the `training`/`inference` services.

```py
from caikit.core import DataObjectBase, ModuleBase, TaskBase, dataobject, module, task
from caikit.core.modules import ModuleLoader, ModuleSaver
from caikit.interfaces.runtime.data_model import TrainingInfoRequest, TrainingStatus
from caikit.runtime import grpc_server
import caikit.config
import os


@dataobject
class Greeting(DataObjectBase):
    greeting: str


@task(unary_parameters={"name": str}, unary_output_type=Greeting)
class GreetingTask(TaskBase):
    pass


@module("greeter", "Sample Greeter", "0.0.0", task=GreetingTask)
class GreeterModule(ModuleBase):

    def __init__(self, greeting_template: str = "Hello {}"):
        self._greeting_template = greeting_template

    def run(self, name: str) -> Greeting:
        return Greeting(self._greeting_template.format(name))

    @classmethod
    def train(cls, greeting_prefix: str) -> "GreeterModule":
        return cls(f"{greeting_prefix} {{}}")

    def save(self, model_path: str):
        with ModuleSaver(module=self, model_path=model_path) as saver:
            saver.update_config({"greeting_template": self._greeting_template})

    @classmethod
    def load(cls, model_path: str) -> "GreeterModule":
        return cls(ModuleLoader(model_path).config.greeting_template)


caikit.configure(
    config_dict={
        "runtime": {
            # Import modules from this script
            "library": "__main__",
            # Auto-load models found in the local "models" directory
            "local_models_dir": "models",
            "lazy_load_local_models": True,
            "training": {
                # Save trained models in the local "models" directory
                "output_dir": "models",
                # Don't save with the model ID for ease of auto-loading
                "save_with_id": False,
            }
        }
    }
)
os.makedirs("models", exist_ok=True)

with grpc_server.RuntimeGRPCServer() as server:
    # Set up service clients
    chan = server.make_local_channel()
    train_client = server.training_service.stub_class(chan)
    train_status_client = server.training_management_service.stub_class(chan)
    inference_client = server.inference_service.stub_class(chan)

    # Launch a training
    training_request = server.training_service.messages.GreetingTaskGreeterModuleTrainRequest(
        model_name="greeter",
        parameters=server.training_service.messages.GreetingTaskGreeterModuleTrainParameters(
            greeting_prefix="Greetings",
        )
    )
    training_handle = train_client.GreetingTaskGreeterModuleTrain(training_request)
    print(f"Started Training {training_handle.training_id} for model {training_handle.model_name}")

    # Wait until the training completes
    while True:
        training_status = train_status_client.GetTrainingStatus(
            TrainingInfoRequest(training_handle.training_id).to_proto()
        )
        if training_status.state == TrainingStatus.COMPLETED.value:
            print(f"Finished training {training_handle.training_id}")
            break

    # Make an inference request
    inference_request = server.inference_service.messages.GreetingTaskRequest(name="Gabe")
    greeting = inference_client.GreetingTaskPredict(
        inference_request, metadata=list({"mm-model-id": training_handle.model_name}.items())
    )
    print(f"Got greeting: {greeting.greeting}")
```

### HTTP Server

For those that prefer REST/HTTP to `grpc`, `caikit` also exposes an HTTP server that can server the same functionality as the `grpc` server in the `caikit.runtime.http_server` module.

```py
from caikit.core import DataObjectBase, ModuleBase, TaskBase, dataobject, module, task
from caikit.core.modules import ModuleLoader, ModuleSaver
from caikit.runtime import http_server
import caikit.config
import os
import requests


@dataobject
class Greeting(DataObjectBase):
    greeting: str


@task(unary_parameters={"name": str}, unary_output_type=Greeting)
class GreetingTask(TaskBase):
    pass


@module("greeter", "Sample Greeter", "0.0.0", task=GreetingTask)
class GreeterModule(ModuleBase):

    def __init__(self, greeting_template: str = "Hello {}"):
        self._greeting_template = greeting_template

    def run(self, name: str) -> Greeting:
        return Greeting(self._greeting_template.format(name))

    @classmethod
    def train(cls, greeting_prefix: str) -> "GreeterModule":
        return cls(f"{greeting_prefix} {{}}")

    def save(self, model_path: str):
        with ModuleSaver(module=self, model_path=model_path) as saver:
            saver.update_config({"greeting_template": self._greeting_template})

    @classmethod
    def load(cls, model_path: str) -> "GreeterModule":
        return cls(ModuleLoader(model_path).config.greeting_template)


caikit.configure(
    config_dict={
        "runtime": {
            # Import modules from this script
            "library": "__main__",
            # Auto-load models found in the local "models" directory
            "local_models_dir": "models",
            "lazy_load_local_models": True,
            "lazy_load_poll_period_seconds": 1,
            "training": {
                # Save trained models in the local "models" directory
                "output_dir": "models",
                # Don't save with the model ID for ease of auto-loading
                "save_with_id": False,
            }
        }
    }
)
os.makedirs("models", exist_ok=True)

with http_server.RuntimeHTTPServer() as server:
    # Set up service clients
    base_url = f"http://localhost:{server.port}"
    train_url = f"{base_url}/api/v1/GreetingTaskGreeterModuleTrain"
    model_info_url = f"{base_url}/info/models"
    inference_url = f"{base_url}/api/v1/task/greeting"

    # Launch a training
    model_name = "greeter"
    training_handle = requests.post(
        train_url, json={"model_name": model_name, "parameters": {"greeting_prefix": "Heyo"}}
    ).json()
    training_id = training_handle["training_id"]
    print(f"Started Training {training_id} for model {model_name}")

    # Wait until the training completes
    # NOTE: Training management not yet available in HTTP server
    while True:
        model_info = requests.get(model_info_url).json()
        if any(model["name"] == model_name for model in model_info["models"]):
            print(f"Finished training {training_id}")
            break

    # Make an inference request
    greeting = requests.post(
        inference_url, json={"inputs": "Gabe", "model_id": model_name},
    ).json()["greeting"]
    print(f"Got greeting: {greeting}")
```

### Model Mesh

In addition to acting as a standalone server, the [gRPC Server](#grpc-server) implements the [`kserve` ModelMesh Sering](https://kserve.github.io/website/0.8/modelserving/mms/modelmesh/overview/) interface as a [ServingRuntime](https://kserve.github.io/website/0.8/modelserving/servingruntimes/).

## 4. AI Domain Interfaces

The `caikit.interfaces` module is the home of concrete AI data structures and `task` definitions. These definitions act as a taxonomy of domains and problems within those domains to help standardize interfaces across implementations in derived libraries. Generally speaking, a `domain` is defined by the standard input data type for a group of problems using the canonical acadamic name (e.g. `nlp` for text-based problems, `vision` for image-based problems). There are certainly problems which span domains and/or don't align with the semantic meaning of the academic name (e.g. code generation which is text based, but not natural language).

### Domain Data Model

Each sub module within `caikit.interfaces.<domain>` holds a `data_model` module that defines the key [dataobject](#data-modeling) structs for the tasks within the given domain.

### Domain Tasks

Each sub module within `caikit.interfaces.<domain>` holds a `tasks` module that defines the key [task signatures](#tasks) for the problems to be solved in the given domain.

## 5. AI Domain Libraries

Within the `caikit` project, there is an evolving set of libraries that offer concrete implementations of [tasks](#tasks) based on their domains.

### Caikit NLP

The most mature AI domain in the `caikit` community is Natural Language Processing (NLP). The [caikit-nlp project](https://github.com/caikit/caikit-nlp) holds modules implementing the most essential tasks in NLP, particular for generative AI usecases.

* [**Text Generation**](https://github.com/caikit/caikit-nlp/tree/main/caikit_nlp/modules/text_generation): Generative models which take prompt text in and generate novel text based on the prompt.
* [**Text Embedding**](https://github.com/caikit/caikit-nlp/tree/main/caikit_nlp/modules/text_embedding): Given input prompt text, generate embedding vectors based on a language model
* [**Tokenization**](https://github.com/caikit/caikit-nlp/tree/main/caikit_nlp/modules/tokenization): Given a string of text, split the text into a sequence of individual tokens
* [**Token Classification**](https://github.com/caikit/caikit-nlp/tree/main/caikit_nlp/modules/token_classification): Given a sequence of tokens, detect token groups which match certain classes (e.g. Personally Identifiable Information)
* [**Text Classification**](https://github.com/caikit/caikit-nlp/tree/main/caikit_nlp/modules/text_classification): Given a string of text, classify it based on a known set of classes

### Caikit Computer Vision

The `caikit-computer-vision` is in its early stages. It currently holds a single `transformers` implementation of the [**Object Detection**](https://github.com/caikit/caikit-computer-vision/tree/main/caikit_computer_vision/modules/object_detection) task, but will continue to grow as usage and contributions add up.

## 6. Prebuilt Runtime Images

The `caikit` community, in close partnership with Red Hat's OpenShift AI team, provides several ways to build/consume container images that are ready to use in a runtime of the user's choosing.

### Caikit NLP

The `caikit-nlp` project provides a [Dockerfile](https://github.com/caikit/caikit-nlp/blob/main/Dockerfile) that builds an image which can be launched to start the `caikit.runtime` server(s) with the `caikit_nlp` modules configured.

### Caikit TGIS Serving

The [`caikit-tgis-serving` project](https://github.com/opendatahub-io/caikit-tgis-serving) from Red Hat OpenShift AI encapsulates one of the most common usage patterns for `caikit`: Efficient serving of Large Language Models. The image combines `caikit`, `caikit-nlp`, and `TGIS` (IBM's [Text Generation Inference Server](https://github.com/opendatahub-io/text-generation-inference)) into a single image that can be run as a [Kserve](https://github.com/opendatahub-io/kserve) `ServingRuntime`.

The prebuilt images can be pulled directly from Red Hat's [quay.io registry](https://quay.io/repository/opendatahub/caikit-tgis-serving).

## 7. Kubernetes Runtime Stack

The highest-level architecture in the `caikit` project is the end-to-end runtime stack running in `kubernetes`. An end-to-end [operator](https://kubernetes.io/docs/concepts/extend-kubernetes/operator/) is currently under development, so check back soon!
