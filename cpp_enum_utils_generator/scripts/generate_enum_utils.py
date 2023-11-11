#!/usr/bin/env python3

import xml.etree.ElementTree as ET
from argparse import ArgumentParser


def generate_iterator_method(enum_spec):
    main_template = '''
/// @brief Calls the specified functor for all __ENUM_NAME__ values
/// @tparam Functor Automatically deduced type of the functor object
/// @param f The functor to call
[[maybe_unused]] template <typename Functor>
constexpr void forAll__ENUM_NAME__(Functor && f) noexcept
{
    __CALLS__
}
    '''
    
    call_template = '''
    f(__VALUE__);'''
    call_string = ''.join([call_template.replace('__VALUE__', val) for val in enum_spec['values']])
    full_body = main_template.replace('__CALLS__', call_string).replace('__ENUM_NAME__', enum_spec['name'])
    print(full_body)


def generate_cast_from_underlying_type(enum_spec):
    main_template = '''
/// @brief Converts the raw data to __ENUM_NAME__ value
/// @param value The value to convert
/// @return __ENUM_NAME__ with the same numeric value as @p value, or empty std::optional object if the @p value is outside valid enum range
[[maybe_unused]] static std::optional<__ENUM_NAME__> fromRawData(const std::underlying_type<__ENUM_NAME__>::type value) noexcept
{
    switch (value)
    {
        __CASES__
            return static_cast<__ENUM_NAME__>(value);
        default:
            return std::nullopt;
    }
}
    '''
    
    case_template = '''
        case __ENUM_NAME__::__VALUE__:'''
    
    __CASES___string = "".join([case_template.replace('__VALUE__', val) for val in enum_spec['values']])
    full_body = main_template.replace('__CASES__', __CASES___string).replace('__ENUM_NAME__', enum_spec['name'])
    print(full_body)


def generate_to_string(enum_spec):
    main_template = '''
/// @brief Converts the enumeration to std::string
/// @param value The value to convert
/// @return String representation of the @p value
[[maybe_unused]] static std::string toString(const __ENUM_NAME__ value) noexcept
{
    switch (value)
    {
        __CASES__
        default:
            return {};
    }
}
    '''
    case_template = '''
        case __ENUM_NAME__::__VALUE__:
            return "__VALUE__";'''
    
    __CASES___string = "".join([case_template.replace('__VALUE__', val) for val in enum_spec['values']])
    full_body = main_template.replace('__CASES__', __CASES___string).replace('__ENUM_NAME__', enum_spec['name'])
    print(full_body)


def generate_string_utils(enum_spec):
    generate_to_string(enum_spec)


def generate_to_string_test_cases(enum_spec):
    main_template = '''
static void test_toString__ENUM_NAME__() noexcept
{   __CASES__
}
    '''
    test_case_template = '''
    ASSERT_EQ(toString(__ENUM_NAME__::__VALUE__), "__VALUE__");'''
    call_string = ''.join([test_case_template.replace('__VALUE__', val) for val in enum_spec['values']])
    full_body = main_template.replace('__CASES__', call_string).replace('__ENUM_NAME__', enum_spec['name'])
    print(full_body)
    

def generate_cast_test_cases(enum_spec):
    main_template = '''
static void test_fromRawData__ENUM_NAME__() noexcept
{   
    using raw_type = std::underlying_type<__ENUM_NAME__>::type;
    __CASES__
}
    '''
    test_case_template = '''
    ASSERT_EQ(fromRawData(static_cast<raw_type>(__ENUM_NAME__::__VALUE__)), __ENUM_NAME__::__VALUE__);'''
    call_string = ''.join([test_case_template.replace('__VALUE__', val) for val in enum_spec['values']])
    call_string += '\n\tASSERT_EQ(fromRawData(static_cast<raw_type>(__ENUM_NAME__::__LAST_VALUE__) + 1), std::nullopt);'
    full_body = main_template.replace('__CASES__', call_string).replace('__ENUM_NAME__', enum_spec['name']).replace('__LAST_VALUE__', enum_spec['values'][-1])
    print(full_body)


def generate_iterator_method_test_cases(enum_spec):
    main_template = '''
static void test_iteration_method___ENUM_NAME__() noexcept
{   
    std::vector<__ENUM_NAME__> call_sequence;
    auto functor = [&call_sequence](const __ENUM_NAME__ val)
    {
        call_sequence.push_back(val);
    };
    forAll__ENUM_NAME__(functor);
    
    ASSERT_EQ(call_sequence.size(), __ENUM_VALUE_COUNT__);
    __CASES__
}
    '''
    test_cast_template = '''
    ASSERT_EQ(call_sequence[__IDX__], __ENUM_NAME__::__VALUE__);'''
    call_string = ''.join([test_cast_template.replace('__VALUE__', val).replace('__IDX__', str(idx)) for idx, val in enumerate(enum_spec['values'])])
    full_body = main_template.replace('__CASES__', call_string).replace('__ENUM_NAME__', enum_spec['name']).replace('__ENUM_VALUE_COUNT__', str(len(enum_spec['values'])))
    print(full_body)


def generate_test_cases(enum_spec):
    generate_to_string_test_cases(enum_spec)
    generate_cast_test_cases(enum_spec)
    generate_iterator_method_test_cases(enum_spec)


def process_enum(enum_tag):
    enum_def = {}
    enum_def['name'] = enum_tag.get('name')
    enum_def['values'] = [val.get('name') for val in enum_tag.findall('EnumValue')]
    if enum_def['values']:
        generate_string_utils(enum_def)
        generate_cast_from_underlying_type(enum_def)
        generate_iterator_method(enum_def)
        generate_test_cases(enum_def)
    else:
        print("// {}: empty enumeration, nothing to generate!".format(enum_def['name']))
    

if __name__ == "__main__":
    parser = ArgumentParser(
                    prog='Enum utils generator',
                    description='Generates utility methods (with test cases) for C++ enums')
    
    parser.add_argument('filename')
    args = parser.parse_args()
    
    root = ET.parse(args.filename).getroot()
    for enum_tag in root.findall('Enumeration'):
        process_enum(enum_tag)
    