
/// @brief Converts the enumeration to std::string
/// @param value The value to convert
/// @return String representation of the @p value
[[maybe_unused]] static std::string toString(const TestEnum value) noexcept
{
    switch (value)
    {
        
        case TestEnum::SUCCESS:
            return "SUCCESS";
        case TestEnum::FAILURE:
            return "FAILURE";
        case TestEnum::VAL_2:
            return "VAL_2";
        case TestEnum::VAL_3:
            return "VAL_3";
        case TestEnum::SIGNAL_4:
            return "SIGNAL_4";
        default:
            return {};
    }
}
    

/// @brief Converts the raw data to TestEnum value
/// @param value The value to convert
/// @return TestEnum with the same numeric value as @p value, or empty std::optional object if the @p value is outside valid enum range
[[maybe_unused]] static std::optional<TestEnum> fromRawData(const std::underlying_type<TestEnum>::type value) noexcept
{
    switch (value)
    {
        
        case TestEnum::SUCCESS:
        case TestEnum::FAILURE:
        case TestEnum::VAL_2:
        case TestEnum::VAL_3:
        case TestEnum::SIGNAL_4:
            return static_cast<TestEnum>(value);
        default:
            return std::nullopt;
    }
}
    

/// @brief Calls the specified functor for all TestEnum values
/// @tparam Functor Automatically deduced type of the functor object
/// @param f The functor to call
[[maybe_unused]] template <typename Functor>
constexpr void forAllTestEnum(Functor && f) noexcept
{
    
    f(SUCCESS);
    f(FAILURE);
    f(VAL_2);
    f(VAL_3);
    f(SIGNAL_4);
}
    

static void test_toStringTestEnum() noexcept
{   
    ASSERT_EQ(toString(TestEnum::SUCCESS), "SUCCESS");
    ASSERT_EQ(toString(TestEnum::FAILURE), "FAILURE");
    ASSERT_EQ(toString(TestEnum::VAL_2), "VAL_2");
    ASSERT_EQ(toString(TestEnum::VAL_3), "VAL_3");
    ASSERT_EQ(toString(TestEnum::SIGNAL_4), "SIGNAL_4");
}
    

static void test_fromRawDataTestEnum() noexcept
{   
    using raw_type = std::underlying_type<TestEnum>::type;
    
    ASSERT_EQ(fromRawData(static_cast<raw_type>(TestEnum::SUCCESS)), TestEnum::SUCCESS);
    ASSERT_EQ(fromRawData(static_cast<raw_type>(TestEnum::FAILURE)), TestEnum::FAILURE);
    ASSERT_EQ(fromRawData(static_cast<raw_type>(TestEnum::VAL_2)), TestEnum::VAL_2);
    ASSERT_EQ(fromRawData(static_cast<raw_type>(TestEnum::VAL_3)), TestEnum::VAL_3);
    ASSERT_EQ(fromRawData(static_cast<raw_type>(TestEnum::SIGNAL_4)), TestEnum::SIGNAL_4);
	ASSERT_EQ(fromRawData(static_cast<raw_type>(TestEnum::SIGNAL_4) + 1), std::nullopt);
}
    

static void test_iteration_method_TestEnum() noexcept
{   
    std::vector<TestEnum> call_sequence;
    auto functor = [&call_sequence](const TestEnum val)
    {
        call_sequence.push_back(val);
    };
    forAllTestEnum(functor);
    
    ASSERT_EQ(call_sequence.size(), 5);
    
    ASSERT_EQ(call_sequence[0], TestEnum::SUCCESS);
    ASSERT_EQ(call_sequence[1], TestEnum::FAILURE);
    ASSERT_EQ(call_sequence[2], TestEnum::VAL_2);
    ASSERT_EQ(call_sequence[3], TestEnum::VAL_3);
    ASSERT_EQ(call_sequence[4], TestEnum::SIGNAL_4);
}
    
