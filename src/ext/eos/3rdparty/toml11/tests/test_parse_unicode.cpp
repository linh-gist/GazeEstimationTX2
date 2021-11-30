#define BOOST_TEST_MODULE "test_parse_unicode"
#ifdef UNITTEST_FRAMEWORK_LIBRARY_EXIST
#include <boost/test/unit_test.hpp>
#else
#define BOOST_TEST_NO_LIB
#include <boost/test/included/unit_test.hpp>
#endif
#include <toml.hpp>
#include <iostream>
#include <fstream>

#if defined(_MSC_VER) || defined(__INTEL_COMPILER)
BOOST_AUTO_TEST_CASE(test_hard_example_unicode)
{
    const auto data = toml::parse("toml/tests/hard_example_unicode.toml");

    const auto the = toml::get<toml::Table>(data.at("the"));
    BOOST_CHECK_EQUAL(toml::get<std::string>(the.at("test_string")),
                      std::string("\xC3\x9D\xC3\xB4\xC3\xBA\x27\xE2\x84\x93\xE2\x84\x93\x20\xCE\xBB\xC3\xA1\xC6\xAD\xC3\xA8\x20\xE2\x82\xA5\xC3\xA8\x20\xC3\xA1\xC6\x92\xC6\xAD\xC3\xA8\xC5\x99\x20\xC6\xAD\xCE\xBB\xC3\xAF\xC6\xA8\x20\x2D\x20\x23"));

    const auto hard = toml::get<toml::Table>(the.at("hard"));
    const std::vector<std::string> expected_the_hard_test_array{"] ", " # "};
    BOOST_CHECK(toml::get<std::vector<std::string>>(hard.at("test_array")) ==
                expected_the_hard_test_array);
    const std::vector<std::string> expected_the_hard_test_array2{
        std::string("\x54\xC3\xA8\xC6\xA8\xC6\xAD\x20\x23\x31\x31\x20\x5D\xC6\xA5\xC5\x99\xC3\xB4\xC6\xB2\xC3\xA8\xCE\xB4\x20\xC6\xAD\xCE\xBB\xC3\xA1\xC6\xAD"),
        std::string("\xC3\x89\xD0\xB6\xC6\xA5\xC3\xA8\xC5\x99\xC3\xAF\xE2\x82\xA5\xC3\xA8\xC3\xB1\xC6\xAD\x20\x23\x39\x20\xCF\x89\xC3\xA1\xC6\xA8\x20\xC3\xA1\x20\xC6\xA8\xC3\xBA\xC3\xA7\xC3\xA7\xC3\xA8\xC6\xA8\xC6\xA8")
    };
    BOOST_CHECK(toml::get<std::vector<std::string>>(hard.at("test_array2")) ==
                expected_the_hard_test_array2);
    BOOST_CHECK_EQUAL(toml::get<std::string>(hard.at("another_test_string")),
                      std::string("\xC2\xA7\xC3\xA1\xE2\x82\xA5\xC3\xA8\x20\xC6\xAD\xCE\xBB\xC3\xAF\xC3\xB1\xCF\xB1\x2C\x20\xCE\xB2\xC3\xBA\xC6\xAD\x20\xCF\x89\xC3\xAF\xC6\xAD\xCE\xBB\x20\xC3\xA1\x20\xC6\xA8\xC6\xAD\xC5\x99\xC3\xAF\xC3\xB1\xCF\xB1\x20\x23"));
    BOOST_CHECK_EQUAL(toml::get<std::string>(hard.at("harder_test_string")),
                      std::string("\x20\xC3\x82\xC3\xB1\xCE\xB4\x20\xCF\x89\xCE\xBB\xC3\xA8\xC3\xB1\x20\x22\x27\xC6\xA8\x20\xC3\xA1\xC5\x99\xC3\xA8\x20\xC3\xAF\xC3\xB1\x20\xC6\xAD\xCE\xBB\xC3\xA8\x20\xC6\xA8\xC6\xAD\xC5\x99\xC3\xAF\xC3\xB1\xCF\xB1\x2C\x20\xC3\xA1\xE2\x84\x93\xC3\xB4\xC3\xB1\xCF\xB1\x20\xCF\x89\xC3\xAF\xC6\xAD\xCE\xBB\x20\x23\x20\x22"));
//
    const auto bit = toml::get<toml::Table>(hard.at(std::string("\xCE\xB2\xC3\xAF\xC6\xAD\x23")));
    BOOST_CHECK_EQUAL(toml::get<std::string>(bit.at(std::string("\xCF\x89\xCE\xBB\xC3\xA1\xC6\xAD\x3F"))),
        std::string("\xC3\x9D\xC3\xB4\xC3\xBA\x20\xCE\xB4\xC3\xB4\xC3\xB1\x27\xC6\xAD\x20\xC6\xAD\xCE\xBB\xC3\xAF\xC3\xB1\xC6\x99\x20\xC6\xA8\xC3\xB4\xE2\x82\xA5\xC3\xA8\x20\xC3\xBA\xC6\xA8\xC3\xA8\xC5\x99\x20\xCF\x89\xC3\xB4\xC3\xB1\x27\xC6\xAD\x20\xCE\xB4\xC3\xB4\x20\xC6\xAD\xCE\xBB\xC3\xA1\xC6\xAD\x3F"));
    const std::vector<std::string> expected_multi_line_array{"]"};
    BOOST_CHECK(toml::get<std::vector<std::string>>(bit.at("multi_line_array")) ==
                expected_multi_line_array);
}
#else
BOOST_AUTO_TEST_CASE(test_hard_example_unicode)
{
    const auto data = toml::parse("toml/tests/hard_example_unicode.toml");

    const auto the = toml::get<toml::Table>(data.at("the"));
    BOOST_CHECK_EQUAL(toml::get<std::string>(the.at("test_string")),
                      std::string(u8"Ýôú'ℓℓ λáƭè ₥è áƒƭèř ƭλïƨ - #"));

    const auto hard = toml::get<toml::Table>(the.at("hard"));
    const std::vector<std::string> expected_the_hard_test_array{"] ", " # "};
    BOOST_CHECK(toml::get<std::vector<std::string>>(hard.at("test_array")) ==
                expected_the_hard_test_array);
    const std::vector<std::string> expected_the_hard_test_array2{
            std::string(u8"Tèƨƭ #11 ]ƥřôƲèδ ƭλáƭ"),
            std::string(u8"Éжƥèřï₥èñƭ #9 ωáƨ á ƨúççèƨƨ")};
    BOOST_CHECK(toml::get<std::vector<std::string>>(hard.at("test_array2")) ==
                expected_the_hard_test_array2);
    BOOST_CHECK_EQUAL(toml::get<std::string>(hard.at("another_test_string")),
                      std::string(u8"§á₥è ƭλïñϱ, βúƭ ωïƭλ á ƨƭřïñϱ #"));
    BOOST_CHECK_EQUAL(toml::get<std::string>(hard.at("harder_test_string")),
                      std::string(u8" Âñδ ωλèñ \"'ƨ ářè ïñ ƭλè ƨƭřïñϱ, áℓôñϱ ωïƭλ # \""));

    const auto bit = toml::get<toml::Table>(hard.at(std::string(u8"βïƭ#")));
    BOOST_CHECK_EQUAL(toml::get<std::string>(bit.at(std::string(u8"ωλáƭ?"))),
                      std::string(u8"Ýôú δôñ'ƭ ƭλïñƙ ƨô₥è úƨèř ωôñ'ƭ δô ƭλáƭ?"));
    const std::vector<std::string> expected_multi_line_array{"]"};
    BOOST_CHECK(toml::get<std::vector<std::string>>(bit.at("multi_line_array")) ==
                expected_multi_line_array);

}
#endif