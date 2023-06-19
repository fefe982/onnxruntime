// automatically generated by the FlatBuffers compiler, do not modify


#ifndef FLATBUFFERS_GENERATED_ORTTRAININGCHECKPOINT_ONNXRUNTIME_FBS_H_
#define FLATBUFFERS_GENERATED_ORTTRAININGCHECKPOINT_ONNXRUNTIME_FBS_H_

#include "flatbuffers/flatbuffers.h"

#include "ort.fbs.h"

namespace onnxruntime {
namespace fbs {

struct ModuleState;
struct ModuleStateBuilder;

struct ParameterOptimizerState;
struct ParameterOptimizerStateBuilder;

struct OptimizerGroup;
struct OptimizerGroupBuilder;

struct IntProperty;
struct IntPropertyBuilder;

struct FloatProperty;
struct FloatPropertyBuilder;

struct StringProperty;
struct StringPropertyBuilder;

struct PropertyBag;
struct PropertyBagBuilder;

struct Checkpoint;
struct CheckpointBuilder;

struct ModuleState FLATBUFFERS_FINAL_CLASS : private flatbuffers::Table {
  typedef ModuleStateBuilder Builder;
  enum FlatBuffersVTableOffset FLATBUFFERS_VTABLE_UNDERLYING_TYPE {
    VT_REQUIRES_GRAD = 4,
    VT_FROZEN_PARAMS = 6
  };
  const flatbuffers::Vector<flatbuffers::Offset<onnxruntime::fbs::Tensor>> *requires_grad() const {
    return GetPointer<const flatbuffers::Vector<flatbuffers::Offset<onnxruntime::fbs::Tensor>> *>(VT_REQUIRES_GRAD);
  }
  const flatbuffers::Vector<flatbuffers::Offset<onnxruntime::fbs::Tensor>> *frozen_params() const {
    return GetPointer<const flatbuffers::Vector<flatbuffers::Offset<onnxruntime::fbs::Tensor>> *>(VT_FROZEN_PARAMS);
  }
  bool Verify(flatbuffers::Verifier &verifier) const {
    return VerifyTableStart(verifier) &&
           VerifyOffset(verifier, VT_REQUIRES_GRAD) &&
           verifier.VerifyVector(requires_grad()) &&
           verifier.VerifyVectorOfTables(requires_grad()) &&
           VerifyOffset(verifier, VT_FROZEN_PARAMS) &&
           verifier.VerifyVector(frozen_params()) &&
           verifier.VerifyVectorOfTables(frozen_params()) &&
           verifier.EndTable();
  }
};

struct ModuleStateBuilder {
  typedef ModuleState Table;
  flatbuffers::FlatBufferBuilder &fbb_;
  flatbuffers::uoffset_t start_;
  void add_requires_grad(flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<onnxruntime::fbs::Tensor>>> requires_grad) {
    fbb_.AddOffset(ModuleState::VT_REQUIRES_GRAD, requires_grad);
  }
  void add_frozen_params(flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<onnxruntime::fbs::Tensor>>> frozen_params) {
    fbb_.AddOffset(ModuleState::VT_FROZEN_PARAMS, frozen_params);
  }
  explicit ModuleStateBuilder(flatbuffers::FlatBufferBuilder &_fbb)
        : fbb_(_fbb) {
    start_ = fbb_.StartTable();
  }
  ModuleStateBuilder &operator=(const ModuleStateBuilder &);
  flatbuffers::Offset<ModuleState> Finish() {
    const auto end = fbb_.EndTable(start_);
    auto o = flatbuffers::Offset<ModuleState>(end);
    return o;
  }
};

inline flatbuffers::Offset<ModuleState> CreateModuleState(
    flatbuffers::FlatBufferBuilder &_fbb,
    flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<onnxruntime::fbs::Tensor>>> requires_grad = 0,
    flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<onnxruntime::fbs::Tensor>>> frozen_params = 0) {
  ModuleStateBuilder builder_(_fbb);
  builder_.add_frozen_params(frozen_params);
  builder_.add_requires_grad(requires_grad);
  return builder_.Finish();
}

inline flatbuffers::Offset<ModuleState> CreateModuleStateDirect(
    flatbuffers::FlatBufferBuilder &_fbb,
    const std::vector<flatbuffers::Offset<onnxruntime::fbs::Tensor>> *requires_grad = nullptr,
    const std::vector<flatbuffers::Offset<onnxruntime::fbs::Tensor>> *frozen_params = nullptr) {
  auto requires_grad__ = requires_grad ? _fbb.CreateVector<flatbuffers::Offset<onnxruntime::fbs::Tensor>>(*requires_grad) : 0;
  auto frozen_params__ = frozen_params ? _fbb.CreateVector<flatbuffers::Offset<onnxruntime::fbs::Tensor>>(*frozen_params) : 0;
  return onnxruntime::fbs::CreateModuleState(
      _fbb,
      requires_grad__,
      frozen_params__);
}

struct ParameterOptimizerState FLATBUFFERS_FINAL_CLASS : private flatbuffers::Table {
  typedef ParameterOptimizerStateBuilder Builder;
  enum FlatBuffersVTableOffset FLATBUFFERS_VTABLE_UNDERLYING_TYPE {
    VT_PARAM_NAME = 4,
    VT_MOMENTUMS = 6
  };
  const flatbuffers::String *param_name() const {
    return GetPointer<const flatbuffers::String *>(VT_PARAM_NAME);
  }
  const flatbuffers::Vector<flatbuffers::Offset<onnxruntime::fbs::Tensor>> *momentums() const {
    return GetPointer<const flatbuffers::Vector<flatbuffers::Offset<onnxruntime::fbs::Tensor>> *>(VT_MOMENTUMS);
  }
  bool Verify(flatbuffers::Verifier &verifier) const {
    return VerifyTableStart(verifier) &&
           VerifyOffset(verifier, VT_PARAM_NAME) &&
           verifier.VerifyString(param_name()) &&
           VerifyOffset(verifier, VT_MOMENTUMS) &&
           verifier.VerifyVector(momentums()) &&
           verifier.VerifyVectorOfTables(momentums()) &&
           verifier.EndTable();
  }
};

struct ParameterOptimizerStateBuilder {
  typedef ParameterOptimizerState Table;
  flatbuffers::FlatBufferBuilder &fbb_;
  flatbuffers::uoffset_t start_;
  void add_param_name(flatbuffers::Offset<flatbuffers::String> param_name) {
    fbb_.AddOffset(ParameterOptimizerState::VT_PARAM_NAME, param_name);
  }
  void add_momentums(flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<onnxruntime::fbs::Tensor>>> momentums) {
    fbb_.AddOffset(ParameterOptimizerState::VT_MOMENTUMS, momentums);
  }
  explicit ParameterOptimizerStateBuilder(flatbuffers::FlatBufferBuilder &_fbb)
        : fbb_(_fbb) {
    start_ = fbb_.StartTable();
  }
  ParameterOptimizerStateBuilder &operator=(const ParameterOptimizerStateBuilder &);
  flatbuffers::Offset<ParameterOptimizerState> Finish() {
    const auto end = fbb_.EndTable(start_);
    auto o = flatbuffers::Offset<ParameterOptimizerState>(end);
    return o;
  }
};

inline flatbuffers::Offset<ParameterOptimizerState> CreateParameterOptimizerState(
    flatbuffers::FlatBufferBuilder &_fbb,
    flatbuffers::Offset<flatbuffers::String> param_name = 0,
    flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<onnxruntime::fbs::Tensor>>> momentums = 0) {
  ParameterOptimizerStateBuilder builder_(_fbb);
  builder_.add_momentums(momentums);
  builder_.add_param_name(param_name);
  return builder_.Finish();
}

inline flatbuffers::Offset<ParameterOptimizerState> CreateParameterOptimizerStateDirect(
    flatbuffers::FlatBufferBuilder &_fbb,
    const char *param_name = nullptr,
    const std::vector<flatbuffers::Offset<onnxruntime::fbs::Tensor>> *momentums = nullptr) {
  auto param_name__ = param_name ? _fbb.CreateString(param_name) : 0;
  auto momentums__ = momentums ? _fbb.CreateVector<flatbuffers::Offset<onnxruntime::fbs::Tensor>>(*momentums) : 0;
  return onnxruntime::fbs::CreateParameterOptimizerState(
      _fbb,
      param_name__,
      momentums__);
}

struct OptimizerGroup FLATBUFFERS_FINAL_CLASS : private flatbuffers::Table {
  typedef OptimizerGroupBuilder Builder;
  enum FlatBuffersVTableOffset FLATBUFFERS_VTABLE_UNDERLYING_TYPE {
    VT_GROUP_NAME = 4,
    VT_STEP = 6,
    VT_INITIAL_LEARNING_RATE = 8,
    VT_OPTIMIZER_STATES = 10
  };
  const flatbuffers::String *group_name() const {
    return GetPointer<const flatbuffers::String *>(VT_GROUP_NAME);
  }
  int64_t step() const {
    return GetField<int64_t>(VT_STEP, 0);
  }
  float initial_learning_rate() const {
    return GetField<float>(VT_INITIAL_LEARNING_RATE, 0.0f);
  }
  const flatbuffers::Vector<flatbuffers::Offset<onnxruntime::fbs::ParameterOptimizerState>> *optimizer_states() const {
    return GetPointer<const flatbuffers::Vector<flatbuffers::Offset<onnxruntime::fbs::ParameterOptimizerState>> *>(VT_OPTIMIZER_STATES);
  }
  bool Verify(flatbuffers::Verifier &verifier) const {
    return VerifyTableStart(verifier) &&
           VerifyOffset(verifier, VT_GROUP_NAME) &&
           verifier.VerifyString(group_name()) &&
           VerifyField<int64_t>(verifier, VT_STEP) &&
           VerifyField<float>(verifier, VT_INITIAL_LEARNING_RATE) &&
           VerifyOffset(verifier, VT_OPTIMIZER_STATES) &&
           verifier.VerifyVector(optimizer_states()) &&
           verifier.VerifyVectorOfTables(optimizer_states()) &&
           verifier.EndTable();
  }
};

struct OptimizerGroupBuilder {
  typedef OptimizerGroup Table;
  flatbuffers::FlatBufferBuilder &fbb_;
  flatbuffers::uoffset_t start_;
  void add_group_name(flatbuffers::Offset<flatbuffers::String> group_name) {
    fbb_.AddOffset(OptimizerGroup::VT_GROUP_NAME, group_name);
  }
  void add_step(int64_t step) {
    fbb_.AddElement<int64_t>(OptimizerGroup::VT_STEP, step, 0);
  }
  void add_initial_learning_rate(float initial_learning_rate) {
    fbb_.AddElement<float>(OptimizerGroup::VT_INITIAL_LEARNING_RATE, initial_learning_rate, 0.0f);
  }
  void add_optimizer_states(flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<onnxruntime::fbs::ParameterOptimizerState>>> optimizer_states) {
    fbb_.AddOffset(OptimizerGroup::VT_OPTIMIZER_STATES, optimizer_states);
  }
  explicit OptimizerGroupBuilder(flatbuffers::FlatBufferBuilder &_fbb)
        : fbb_(_fbb) {
    start_ = fbb_.StartTable();
  }
  OptimizerGroupBuilder &operator=(const OptimizerGroupBuilder &);
  flatbuffers::Offset<OptimizerGroup> Finish() {
    const auto end = fbb_.EndTable(start_);
    auto o = flatbuffers::Offset<OptimizerGroup>(end);
    return o;
  }
};

inline flatbuffers::Offset<OptimizerGroup> CreateOptimizerGroup(
    flatbuffers::FlatBufferBuilder &_fbb,
    flatbuffers::Offset<flatbuffers::String> group_name = 0,
    int64_t step = 0,
    float initial_learning_rate = 0.0f,
    flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<onnxruntime::fbs::ParameterOptimizerState>>> optimizer_states = 0) {
  OptimizerGroupBuilder builder_(_fbb);
  builder_.add_step(step);
  builder_.add_optimizer_states(optimizer_states);
  builder_.add_initial_learning_rate(initial_learning_rate);
  builder_.add_group_name(group_name);
  return builder_.Finish();
}

inline flatbuffers::Offset<OptimizerGroup> CreateOptimizerGroupDirect(
    flatbuffers::FlatBufferBuilder &_fbb,
    const char *group_name = nullptr,
    int64_t step = 0,
    float initial_learning_rate = 0.0f,
    const std::vector<flatbuffers::Offset<onnxruntime::fbs::ParameterOptimizerState>> *optimizer_states = nullptr) {
  auto group_name__ = group_name ? _fbb.CreateString(group_name) : 0;
  auto optimizer_states__ = optimizer_states ? _fbb.CreateVector<flatbuffers::Offset<onnxruntime::fbs::ParameterOptimizerState>>(*optimizer_states) : 0;
  return onnxruntime::fbs::CreateOptimizerGroup(
      _fbb,
      group_name__,
      step,
      initial_learning_rate,
      optimizer_states__);
}

struct IntProperty FLATBUFFERS_FINAL_CLASS : private flatbuffers::Table {
  typedef IntPropertyBuilder Builder;
  enum FlatBuffersVTableOffset FLATBUFFERS_VTABLE_UNDERLYING_TYPE {
    VT_NAME = 4,
    VT_VALUE = 6
  };
  const flatbuffers::String *name() const {
    return GetPointer<const flatbuffers::String *>(VT_NAME);
  }
  int64_t value() const {
    return GetField<int64_t>(VT_VALUE, 0);
  }
  bool Verify(flatbuffers::Verifier &verifier) const {
    return VerifyTableStart(verifier) &&
           VerifyOffset(verifier, VT_NAME) &&
           verifier.VerifyString(name()) &&
           VerifyField<int64_t>(verifier, VT_VALUE) &&
           verifier.EndTable();
  }
};

struct IntPropertyBuilder {
  typedef IntProperty Table;
  flatbuffers::FlatBufferBuilder &fbb_;
  flatbuffers::uoffset_t start_;
  void add_name(flatbuffers::Offset<flatbuffers::String> name) {
    fbb_.AddOffset(IntProperty::VT_NAME, name);
  }
  void add_value(int64_t value) {
    fbb_.AddElement<int64_t>(IntProperty::VT_VALUE, value, 0);
  }
  explicit IntPropertyBuilder(flatbuffers::FlatBufferBuilder &_fbb)
        : fbb_(_fbb) {
    start_ = fbb_.StartTable();
  }
  IntPropertyBuilder &operator=(const IntPropertyBuilder &);
  flatbuffers::Offset<IntProperty> Finish() {
    const auto end = fbb_.EndTable(start_);
    auto o = flatbuffers::Offset<IntProperty>(end);
    return o;
  }
};

inline flatbuffers::Offset<IntProperty> CreateIntProperty(
    flatbuffers::FlatBufferBuilder &_fbb,
    flatbuffers::Offset<flatbuffers::String> name = 0,
    int64_t value = 0) {
  IntPropertyBuilder builder_(_fbb);
  builder_.add_value(value);
  builder_.add_name(name);
  return builder_.Finish();
}

inline flatbuffers::Offset<IntProperty> CreateIntPropertyDirect(
    flatbuffers::FlatBufferBuilder &_fbb,
    const char *name = nullptr,
    int64_t value = 0) {
  auto name__ = name ? _fbb.CreateString(name) : 0;
  return onnxruntime::fbs::CreateIntProperty(
      _fbb,
      name__,
      value);
}

struct FloatProperty FLATBUFFERS_FINAL_CLASS : private flatbuffers::Table {
  typedef FloatPropertyBuilder Builder;
  enum FlatBuffersVTableOffset FLATBUFFERS_VTABLE_UNDERLYING_TYPE {
    VT_NAME = 4,
    VT_VALUE = 6
  };
  const flatbuffers::String *name() const {
    return GetPointer<const flatbuffers::String *>(VT_NAME);
  }
  float value() const {
    return GetField<float>(VT_VALUE, 0.0f);
  }
  bool Verify(flatbuffers::Verifier &verifier) const {
    return VerifyTableStart(verifier) &&
           VerifyOffset(verifier, VT_NAME) &&
           verifier.VerifyString(name()) &&
           VerifyField<float>(verifier, VT_VALUE) &&
           verifier.EndTable();
  }
};

struct FloatPropertyBuilder {
  typedef FloatProperty Table;
  flatbuffers::FlatBufferBuilder &fbb_;
  flatbuffers::uoffset_t start_;
  void add_name(flatbuffers::Offset<flatbuffers::String> name) {
    fbb_.AddOffset(FloatProperty::VT_NAME, name);
  }
  void add_value(float value) {
    fbb_.AddElement<float>(FloatProperty::VT_VALUE, value, 0.0f);
  }
  explicit FloatPropertyBuilder(flatbuffers::FlatBufferBuilder &_fbb)
        : fbb_(_fbb) {
    start_ = fbb_.StartTable();
  }
  FloatPropertyBuilder &operator=(const FloatPropertyBuilder &);
  flatbuffers::Offset<FloatProperty> Finish() {
    const auto end = fbb_.EndTable(start_);
    auto o = flatbuffers::Offset<FloatProperty>(end);
    return o;
  }
};

inline flatbuffers::Offset<FloatProperty> CreateFloatProperty(
    flatbuffers::FlatBufferBuilder &_fbb,
    flatbuffers::Offset<flatbuffers::String> name = 0,
    float value = 0.0f) {
  FloatPropertyBuilder builder_(_fbb);
  builder_.add_value(value);
  builder_.add_name(name);
  return builder_.Finish();
}

inline flatbuffers::Offset<FloatProperty> CreateFloatPropertyDirect(
    flatbuffers::FlatBufferBuilder &_fbb,
    const char *name = nullptr,
    float value = 0.0f) {
  auto name__ = name ? _fbb.CreateString(name) : 0;
  return onnxruntime::fbs::CreateFloatProperty(
      _fbb,
      name__,
      value);
}

struct StringProperty FLATBUFFERS_FINAL_CLASS : private flatbuffers::Table {
  typedef StringPropertyBuilder Builder;
  enum FlatBuffersVTableOffset FLATBUFFERS_VTABLE_UNDERLYING_TYPE {
    VT_NAME = 4,
    VT_VALUE = 6
  };
  const flatbuffers::String *name() const {
    return GetPointer<const flatbuffers::String *>(VT_NAME);
  }
  const flatbuffers::String *value() const {
    return GetPointer<const flatbuffers::String *>(VT_VALUE);
  }
  bool Verify(flatbuffers::Verifier &verifier) const {
    return VerifyTableStart(verifier) &&
           VerifyOffset(verifier, VT_NAME) &&
           verifier.VerifyString(name()) &&
           VerifyOffset(verifier, VT_VALUE) &&
           verifier.VerifyString(value()) &&
           verifier.EndTable();
  }
};

struct StringPropertyBuilder {
  typedef StringProperty Table;
  flatbuffers::FlatBufferBuilder &fbb_;
  flatbuffers::uoffset_t start_;
  void add_name(flatbuffers::Offset<flatbuffers::String> name) {
    fbb_.AddOffset(StringProperty::VT_NAME, name);
  }
  void add_value(flatbuffers::Offset<flatbuffers::String> value) {
    fbb_.AddOffset(StringProperty::VT_VALUE, value);
  }
  explicit StringPropertyBuilder(flatbuffers::FlatBufferBuilder &_fbb)
        : fbb_(_fbb) {
    start_ = fbb_.StartTable();
  }
  StringPropertyBuilder &operator=(const StringPropertyBuilder &);
  flatbuffers::Offset<StringProperty> Finish() {
    const auto end = fbb_.EndTable(start_);
    auto o = flatbuffers::Offset<StringProperty>(end);
    return o;
  }
};

inline flatbuffers::Offset<StringProperty> CreateStringProperty(
    flatbuffers::FlatBufferBuilder &_fbb,
    flatbuffers::Offset<flatbuffers::String> name = 0,
    flatbuffers::Offset<flatbuffers::String> value = 0) {
  StringPropertyBuilder builder_(_fbb);
  builder_.add_value(value);
  builder_.add_name(name);
  return builder_.Finish();
}

inline flatbuffers::Offset<StringProperty> CreateStringPropertyDirect(
    flatbuffers::FlatBufferBuilder &_fbb,
    const char *name = nullptr,
    const char *value = nullptr) {
  auto name__ = name ? _fbb.CreateString(name) : 0;
  auto value__ = value ? _fbb.CreateString(value) : 0;
  return onnxruntime::fbs::CreateStringProperty(
      _fbb,
      name__,
      value__);
}

struct PropertyBag FLATBUFFERS_FINAL_CLASS : private flatbuffers::Table {
  typedef PropertyBagBuilder Builder;
  enum FlatBuffersVTableOffset FLATBUFFERS_VTABLE_UNDERLYING_TYPE {
    VT_INTS = 4,
    VT_FLOATS = 6,
    VT_STRINGS = 8
  };
  const flatbuffers::Vector<flatbuffers::Offset<onnxruntime::fbs::IntProperty>> *ints() const {
    return GetPointer<const flatbuffers::Vector<flatbuffers::Offset<onnxruntime::fbs::IntProperty>> *>(VT_INTS);
  }
  const flatbuffers::Vector<flatbuffers::Offset<onnxruntime::fbs::FloatProperty>> *floats() const {
    return GetPointer<const flatbuffers::Vector<flatbuffers::Offset<onnxruntime::fbs::FloatProperty>> *>(VT_FLOATS);
  }
  const flatbuffers::Vector<flatbuffers::Offset<onnxruntime::fbs::StringProperty>> *strings() const {
    return GetPointer<const flatbuffers::Vector<flatbuffers::Offset<onnxruntime::fbs::StringProperty>> *>(VT_STRINGS);
  }
  bool Verify(flatbuffers::Verifier &verifier) const {
    return VerifyTableStart(verifier) &&
           VerifyOffset(verifier, VT_INTS) &&
           verifier.VerifyVector(ints()) &&
           verifier.VerifyVectorOfTables(ints()) &&
           VerifyOffset(verifier, VT_FLOATS) &&
           verifier.VerifyVector(floats()) &&
           verifier.VerifyVectorOfTables(floats()) &&
           VerifyOffset(verifier, VT_STRINGS) &&
           verifier.VerifyVector(strings()) &&
           verifier.VerifyVectorOfTables(strings()) &&
           verifier.EndTable();
  }
};

struct PropertyBagBuilder {
  typedef PropertyBag Table;
  flatbuffers::FlatBufferBuilder &fbb_;
  flatbuffers::uoffset_t start_;
  void add_ints(flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<onnxruntime::fbs::IntProperty>>> ints) {
    fbb_.AddOffset(PropertyBag::VT_INTS, ints);
  }
  void add_floats(flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<onnxruntime::fbs::FloatProperty>>> floats) {
    fbb_.AddOffset(PropertyBag::VT_FLOATS, floats);
  }
  void add_strings(flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<onnxruntime::fbs::StringProperty>>> strings) {
    fbb_.AddOffset(PropertyBag::VT_STRINGS, strings);
  }
  explicit PropertyBagBuilder(flatbuffers::FlatBufferBuilder &_fbb)
        : fbb_(_fbb) {
    start_ = fbb_.StartTable();
  }
  PropertyBagBuilder &operator=(const PropertyBagBuilder &);
  flatbuffers::Offset<PropertyBag> Finish() {
    const auto end = fbb_.EndTable(start_);
    auto o = flatbuffers::Offset<PropertyBag>(end);
    return o;
  }
};

inline flatbuffers::Offset<PropertyBag> CreatePropertyBag(
    flatbuffers::FlatBufferBuilder &_fbb,
    flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<onnxruntime::fbs::IntProperty>>> ints = 0,
    flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<onnxruntime::fbs::FloatProperty>>> floats = 0,
    flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<onnxruntime::fbs::StringProperty>>> strings = 0) {
  PropertyBagBuilder builder_(_fbb);
  builder_.add_strings(strings);
  builder_.add_floats(floats);
  builder_.add_ints(ints);
  return builder_.Finish();
}

inline flatbuffers::Offset<PropertyBag> CreatePropertyBagDirect(
    flatbuffers::FlatBufferBuilder &_fbb,
    const std::vector<flatbuffers::Offset<onnxruntime::fbs::IntProperty>> *ints = nullptr,
    const std::vector<flatbuffers::Offset<onnxruntime::fbs::FloatProperty>> *floats = nullptr,
    const std::vector<flatbuffers::Offset<onnxruntime::fbs::StringProperty>> *strings = nullptr) {
  auto ints__ = ints ? _fbb.CreateVector<flatbuffers::Offset<onnxruntime::fbs::IntProperty>>(*ints) : 0;
  auto floats__ = floats ? _fbb.CreateVector<flatbuffers::Offset<onnxruntime::fbs::FloatProperty>>(*floats) : 0;
  auto strings__ = strings ? _fbb.CreateVector<flatbuffers::Offset<onnxruntime::fbs::StringProperty>>(*strings) : 0;
  return onnxruntime::fbs::CreatePropertyBag(
      _fbb,
      ints__,
      floats__,
      strings__);
}

struct Checkpoint FLATBUFFERS_FINAL_CLASS : private flatbuffers::Table {
  typedef CheckpointBuilder Builder;
  enum FlatBuffersVTableOffset FLATBUFFERS_VTABLE_UNDERLYING_TYPE {
    VT_MODULE_STATE = 4,
    VT_OPTIMIZER_GROUPS = 6,
    VT_PROPERTY_BAG = 8,
    VT_VERSION = 10
  };
  const onnxruntime::fbs::ModuleState *module_state() const {
    return GetPointer<const onnxruntime::fbs::ModuleState *>(VT_MODULE_STATE);
  }
  const flatbuffers::Vector<flatbuffers::Offset<onnxruntime::fbs::OptimizerGroup>> *optimizer_groups() const {
    return GetPointer<const flatbuffers::Vector<flatbuffers::Offset<onnxruntime::fbs::OptimizerGroup>> *>(VT_OPTIMIZER_GROUPS);
  }
  const onnxruntime::fbs::PropertyBag *property_bag() const {
    return GetPointer<const onnxruntime::fbs::PropertyBag *>(VT_PROPERTY_BAG);
  }
  int32_t version() const {
    return GetField<int32_t>(VT_VERSION, 0);
  }
  bool Verify(flatbuffers::Verifier &verifier) const {
    return VerifyTableStart(verifier) &&
           VerifyOffset(verifier, VT_MODULE_STATE) &&
           verifier.VerifyTable(module_state()) &&
           VerifyOffset(verifier, VT_OPTIMIZER_GROUPS) &&
           verifier.VerifyVector(optimizer_groups()) &&
           verifier.VerifyVectorOfTables(optimizer_groups()) &&
           VerifyOffset(verifier, VT_PROPERTY_BAG) &&
           verifier.VerifyTable(property_bag()) &&
           VerifyField<int32_t>(verifier, VT_VERSION) &&
           verifier.EndTable();
  }
};

struct CheckpointBuilder {
  typedef Checkpoint Table;
  flatbuffers::FlatBufferBuilder &fbb_;
  flatbuffers::uoffset_t start_;
  void add_module_state(flatbuffers::Offset<onnxruntime::fbs::ModuleState> module_state) {
    fbb_.AddOffset(Checkpoint::VT_MODULE_STATE, module_state);
  }
  void add_optimizer_groups(flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<onnxruntime::fbs::OptimizerGroup>>> optimizer_groups) {
    fbb_.AddOffset(Checkpoint::VT_OPTIMIZER_GROUPS, optimizer_groups);
  }
  void add_property_bag(flatbuffers::Offset<onnxruntime::fbs::PropertyBag> property_bag) {
    fbb_.AddOffset(Checkpoint::VT_PROPERTY_BAG, property_bag);
  }
  void add_version(int32_t version) {
    fbb_.AddElement<int32_t>(Checkpoint::VT_VERSION, version, 0);
  }
  explicit CheckpointBuilder(flatbuffers::FlatBufferBuilder &_fbb)
        : fbb_(_fbb) {
    start_ = fbb_.StartTable();
  }
  CheckpointBuilder &operator=(const CheckpointBuilder &);
  flatbuffers::Offset<Checkpoint> Finish() {
    const auto end = fbb_.EndTable(start_);
    auto o = flatbuffers::Offset<Checkpoint>(end);
    return o;
  }
};

inline flatbuffers::Offset<Checkpoint> CreateCheckpoint(
    flatbuffers::FlatBufferBuilder &_fbb,
    flatbuffers::Offset<onnxruntime::fbs::ModuleState> module_state = 0,
    flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<onnxruntime::fbs::OptimizerGroup>>> optimizer_groups = 0,
    flatbuffers::Offset<onnxruntime::fbs::PropertyBag> property_bag = 0,
    int32_t version = 0) {
  CheckpointBuilder builder_(_fbb);
  builder_.add_version(version);
  builder_.add_property_bag(property_bag);
  builder_.add_optimizer_groups(optimizer_groups);
  builder_.add_module_state(module_state);
  return builder_.Finish();
}

inline flatbuffers::Offset<Checkpoint> CreateCheckpointDirect(
    flatbuffers::FlatBufferBuilder &_fbb,
    flatbuffers::Offset<onnxruntime::fbs::ModuleState> module_state = 0,
    const std::vector<flatbuffers::Offset<onnxruntime::fbs::OptimizerGroup>> *optimizer_groups = nullptr,
    flatbuffers::Offset<onnxruntime::fbs::PropertyBag> property_bag = 0,
    int32_t version = 0) {
  auto optimizer_groups__ = optimizer_groups ? _fbb.CreateVector<flatbuffers::Offset<onnxruntime::fbs::OptimizerGroup>>(*optimizer_groups) : 0;
  return onnxruntime::fbs::CreateCheckpoint(
      _fbb,
      module_state,
      optimizer_groups__,
      property_bag,
      version);
}

inline const onnxruntime::fbs::Checkpoint *GetCheckpoint(const void *buf) {
  return flatbuffers::GetRoot<onnxruntime::fbs::Checkpoint>(buf);
}

inline const onnxruntime::fbs::Checkpoint *GetSizePrefixedCheckpoint(const void *buf) {
  return flatbuffers::GetSizePrefixedRoot<onnxruntime::fbs::Checkpoint>(buf);
}

inline const char *CheckpointIdentifier() {
  return "ODTC";
}

inline bool CheckpointBufferHasIdentifier(const void *buf) {
  return flatbuffers::BufferHasIdentifier(
      buf, CheckpointIdentifier());
}

inline bool VerifyCheckpointBuffer(
    flatbuffers::Verifier &verifier) {
  return verifier.VerifyBuffer<onnxruntime::fbs::Checkpoint>(CheckpointIdentifier());
}

inline bool VerifySizePrefixedCheckpointBuffer(
    flatbuffers::Verifier &verifier) {
  return verifier.VerifySizePrefixedBuffer<onnxruntime::fbs::Checkpoint>(CheckpointIdentifier());
}

inline void FinishCheckpointBuffer(
    flatbuffers::FlatBufferBuilder &fbb,
    flatbuffers::Offset<onnxruntime::fbs::Checkpoint> root) {
  fbb.Finish(root, CheckpointIdentifier());
}

inline void FinishSizePrefixedCheckpointBuffer(
    flatbuffers::FlatBufferBuilder &fbb,
    flatbuffers::Offset<onnxruntime::fbs::Checkpoint> root) {
  fbb.FinishSizePrefixed(root, CheckpointIdentifier());
}

}  // namespace fbs
}  // namespace onnxruntime

#endif  // FLATBUFFERS_GENERATED_ORTTRAININGCHECKPOINT_ONNXRUNTIME_FBS_H_
