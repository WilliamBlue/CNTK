//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include "stdafx.h"
#include "CNTKLibrary.h"
#include <fstream>
#include "Utils.h"

namespace CNTK
{
    struct VariableFields final : public std::enable_shared_from_this<VariableFields>
    {
        friend class CompositeFunction;

        NDShape m_shape;
        VariableKind m_varKind;
        ::CNTK::DataType m_dataType;
        Function* m_ownerFunction; // Variable does not keep the Function alive
        std::unique_ptr<std::once_flag> m_initValueFlag;
        NDArrayViewPtr m_value;
        std::unique_ptr<ParameterInitializer> m_valueInitializer;
        std::unique_ptr<DeviceDescriptor> m_valueInitializationDevice;
        bool m_needsGradient;
        std::wstring m_name;
        std::vector<Axis> m_dynamicAxes;
        bool m_isSparse;
        std::wstring m_uid;
        std::atomic<size_t> m_valueTimeStamp;
        Variable m_blockFunctionVariableMapping;

        VariableFields(const NDShape& shape, VariableKind varType, ::CNTK::DataType type, Function* ownerFunction, const NDArrayViewPtr& value, bool needsGradient, const std::vector<Axis>& dynamicAxes, bool isSparse, const std::wstring& name, const std::wstring& uid)
            : m_shape(shape), m_varKind(varType), m_dataType(type), m_ownerFunction(ownerFunction), m_value(value), m_needsGradient(needsGradient), m_dynamicAxes(dynamicAxes), m_isSparse(isSparse), m_name(name), m_uid(uid), m_valueTimeStamp(0)
        {
            if (value && (type != value->GetDataType()))
                InvalidArgument("The DataType of the Parameter/Constant Variable '%S' does not match the DataType of the associated Value", AsString().c_str());

            // Validate that each of the dynamic axes are unique
            std::unordered_set<Axis> uniqueDynamicAxis;
            for (auto& currentDynamicAxis : dynamicAxes)
            {
                auto retVal = uniqueDynamicAxis.insert(currentDynamicAxis);
                if (!retVal.second)
                    InvalidArgument("Dynamic axis named %S is specified more than once for Variable '%S'", currentDynamicAxis.Name().c_str(), AsString().c_str());
            }
        }

        std::wstring AsString() const
        {
            std::wstringstream wss;
            wss << VariableKindName(m_varKind) << "('";
            if (m_name != L"")
                wss << m_name;
            else
                wss << m_uid;
            bool reverse = Internal::IsReversingTensorShapesInErrorMessagesEnabled();
            if (reverse)
                wss << "', " << DynamicAxesAsString(m_dynamicAxes, reverse) << ", " << m_shape.AsString() << ")";
            else
                wss << "', " << m_shape.AsString() << ", " << DynamicAxesAsString(m_dynamicAxes, reverse) << ")";
            return wss.str();
        }

        std::shared_ptr<VariableFields> Clone() const
        {
            if (m_ownerFunction != nullptr)
                InvalidArgument("Output variable '%S' cannot be cloned.", AsString().c_str());

            // Note: We do not clone m_blockFunctionVariableMapping

            auto clone = MakeSharedObject<VariableFields>(m_shape,
                m_varKind,
                m_dataType,
                m_ownerFunction,
                (m_value) ? m_value->DeepClone() : nullptr,
                m_needsGradient,
                m_dynamicAxes,
                m_isSparse,
                m_name,
                Internal::GenerateUid(m_varKind));

            if (m_valueInitializer)
                clone->SetValueInitialization(*m_valueInitializer, *m_valueInitializationDevice);

            return clone;
        }

        CNTK_API void SetValueInitialization(const ParameterInitializer& initializationConfig, const DeviceDescriptor& device);

    private:
        // Disallow copy and move construction and assignment
        VariableFields(const VariableFields&) = delete; VariableFields& operator=(const VariableFields& other) = delete; VariableFields(VariableFields&&) = delete; VariableFields& operator=(VariableFields&&) = delete;
    };
}
