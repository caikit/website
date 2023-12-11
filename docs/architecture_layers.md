# Architecture Layers

The `caikit` project is itself a stack of complimentary building blocks that layer on top of each other to provide rising levels of abstraction to the various functions in the space of AI problems. The layers roughly break down as follows:

1. [**Useful Tools**](#1-useful-tools): At the bottom, `caikit` holds a collection of tools that are not AI-specific. These tools are useful for AI, but can be imported and used in just about any problem space where they fit.

2. [**AI Data and Model Abstractions**](#2-ai-data-and-model-abstractions): Building on the **Useful Tools**, `caikit` provides a set of abstractions that help to frame AI models based on how a user would consume them (as opposed to how a Data Scientist would author them). Like the **Useful Tools**, these abstractions can be imported and used in any project where there is a need for a pluggable set of runnable objects which can be serialized and configured.

3. [**AI Runtime**](#3-ai-runtime): One of the most common needs in any AI application is a server that can provide API access to performing AI tasks in a repeatable way. These tasks are generally `train` (any form of model creation/customization that results in usable artifacts), and `run` (any form of model inference which takes user input and produces model outputs).

4. [**AI Domain Interfaces**](#4-ai-domain-interfaces): The space of AI is large, and the number of different `domains`, `tasks`, and `data objects` that can be created can be extremely daunting. Many AI platforms attempt to solve this by leaving the interfaces as opaque blobs that only need to be parsable by the model code. This, however, tightly couples the client-side usage of the models to the model implementation which can cause brittleness and migration challenges over time. The `caikit.interfaces` data model attempts to formalize the `domains`, `tasks`, and `data objects` that are most commonly implemented by AI models so that clients can write their code against these interfaces without binding themselves to specific model implementations.

5. [**AI Domain Libraries**](#5-ai-domain-libraries): Outside of the core `caikit` library, the `caikit` community supports an ever-growing set of domain libraries which provide concrete implementations of the tasks defined in `caikit.interfaces`. These libraries provide tested model implementations that can be used out-of-the box (subject to each project's maturity level). They also encode some of the more advanced usecases for optimized runtimes (e.g. [text-generation using a remote TGIS backend](https://github.com/caikit/caikit-nlp/blob/main/caikit_nlp/modules/text_generation/text_generation_tgis.py))

6. [**Prebuilt Runtime Images**](#6-prebuilt-runtime-images): The most common way to encapsulate an instance of an **AI Runtime** is in a [OCI container image](https://opencontainers.org/) (e.g. [docker](https://www.docker.com/), [podman](https://podman.io/), [rancher](https://www.rancher.com/)). Such images can be run using container runtimes such as [kubernetes](https://kubernetes.io/). The `caikit` community provides prebuilt images through our work with Red Hat OpenShift AI which can be directly used and run in a user's application. These images are built using a single **AI Domain Library** and are each capable of running the collection of modules defined there.

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

### Model Management

### Augmentors

## 3. AI Runtime

### gRPC Server

### HTTP Server

## 4. AI Domain Interfaces

### Data Model

### Tasks

## 5. AI Domain Libraries

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