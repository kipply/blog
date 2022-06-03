+++
title = "Tensorflow Serving Client with C++ and Bazel"
date = 2022-02-25
weight = 4
+++

> All code examples can be found in [this repo](https://github.com/kipply/tf-serving-cpp). If you know what you're looking for and can extract the value you need from that repository, the following post may be pretty useless.

Tensorflow doesn't yet have support for TF Serving clients. For Golang, I opted to [compile the protobufs once and put the generated client in the codebase](https://kipp.ly/blog/technical/tf-go/). For a Bazel C++ project, I had two other reasonable options. One was to build tensorflow/serving with my project and use the build target that tensorflow set up for me. I tried that (as another blog post on the internet recommended) and bad things happened. For one, tensorflow/serving relies on tensorflow/tensorflow which has a whole bunch of shit going on (namely long build times) and dependencies on parts of cuda. And two, they had dependencies that clashed with some of the other items in my `WORKSPACE` and it seems that they might still rely on bazel 3.7.0 and I would like to use 5.0.0. So I went with the copy the protos in your project then do the build (which someone else did [for CMake](https://github.com/andrew-k-21-12/tf-serving-client-cpp)).

Reference Links: [gRPC](https://grpc.io/) || [Protocol Buffers C++](https://developers.google.com/protocol-buffers/docs/cpptutorial) || [Tensorflow Serving](https://www.tensorflow.org/tfx/serving/serving_basic) || [Bazel Protocol Buffer Rules](https://docs.bazel.build/versions/main/be/protocol-buffer.html)

1the Protos

You don't need to copy all of the protos in tensorflow/serving and certainly not in tensor/tensorflow as you probably only plan on using the `PredictionService` (as opposed to classify or regress, in which case you'll have to do some of your own proto copying).

Otherwise, these protos can be downloaded from the [git repo here](https://github.com/kipply/tf-serving-cpp), and they're up to date as of tensorflow/serving version 2.8.0 and are very unlikely to change in breaking ways in the near future.

If you put them in a different folder, you'll probably need to rename the imports in the `.proto` files.

### Starlark, star bright, first star I see tonight

The protos don't build themselves, and the compilation setup is 200 lines of Starlark (I'm doing ok here, tensorflow/serving uses 450 lines). copy them [here](https://raw.githubusercontent.com/kipply/tf-serving-cpp/main/BUILD) and leave out the last ten lines that builds a main file.

In your `WORKSPACE` file you'll want the following;

```python
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

#
# GRPC Dependencies
#
http_archive(
    name = "com_github_grpc_grpc",
    strip_prefix = "grpc-dc78581af30da834b7b95572f109bf6c708686e0",
    urls = [
        "https://github.com/grpc/grpc/archive/dc78581af30da834b7b95572f109bf6c708686e0.tar.gz",
    ],
)

load("@com_github_grpc_grpc//bazel:grpc_deps.bzl", "grpc_deps")

grpc_deps()

load("@com_github_grpc_grpc//bazel:grpc_extra_deps.bzl", "grpc_extra_deps")

grpc_extra_deps()

#
# Protocol buffer rules
#
http_archive(
    name = "rules_proto",
    sha256 = "66bfdf8782796239d3875d37e7de19b1d94301e8972b3cbd2446b332429b4df1",
    strip_prefix = "rules_proto-4.0.0",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/rules_proto/archive/refs/tags/4.0.0.tar.gz",
        "https://github.com/bazelbuild/rules_proto/archive/refs/tags/4.0.0.tar.gz",
    ],
)

load("@rules_proto//proto:repositories.bzl", "rules_proto_dependencies", "rules_proto_toolchains")

rules_proto_dependencies()

rules_proto_toolchains()
```

### TF Serving Client over GRPC

I'll leave an example here, but be prepared to reference the C++ GRPC documentation a lot. I personally found it easier to read the header files, Google is quite diligent in leaving lots of comments in there though is not good at turning up the right reference documentation when you search for things.

```c++
#include "grpcpp/grpcpp.h"
#include "tensorflow_serving/apis/prediction_service.grpc.pb.h"

int main(int argc, char** argv) {
  const auto& prediction_service_stub =
      tensorflow::serving::PredictionService::NewStub(grpc::CreateChannel(
          "localhost:9000", grpc::InsecureChannelCredentials())); // make sure your server is running here!

  grpc::ClientContext client_context;
  tensorflow::serving::PredictRequest predict_request;
  tensorflow::serving::PredictResponse predict_response;
  grpc::ClientContext cli_context;

  predict_request.mutable_model_spec()->set_name("model_name"); // specify your model name here
  predict_request.mutable_model_spec()->set_signature_name("some_signature"); // specify the signature here
  google::protobuf::Map<std::string, tensorflow::TensorProto>& inputs =
      *predict_request.mutable_inputs();

  tensorflow::TensorProto input_tensor;
  input_tensor.set_dtype(tensorflow::DataType::DT_INT32);
  input_tensor.mutable_tensor_shape()->add_dim()->set_size(2);
  input_tensor.add_int_val(1);
  input_tensor.add_int_val(2);
  inputs.insert({"input_key", input_tensor});

  const grpc::Status& predict_status = prediction_service_stub->Predict(
      &cli_context, predict_request, &predict_response);

  if (!predict_status.ok()) {
    std::cerr << predict_status.error_message() << std::endl;
    return -1;
  }

  for (const auto& output_pair : predict_response.outputs()) {
    std::cout << "Output " << output_pair.first << std::endl;
    auto tensor = output_pair.second;

    for (const auto val : tensor.int_val()) {
      std::cout << "\t" << val;
    }
    std::cout << std::endl;
  }
}
```

A thing about tensorflow/serving is that you specify dimensions, and you don't model your data as any kind of multi-dimensional tensor. Instead you just `add_dtype_val` until you've filled your specified dimensions. [Here is the link to the full script](https://github.com/kipply/tf-serving-cpp/blob/main/main.cc) which also contains an example to get metadata from tensorflow/serving and is executable in bazel with `bazel run :main_lib`.
