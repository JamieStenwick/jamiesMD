// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUBLASDX_DETAIL_EXPECTED_HPP
#define CUBLASDX_DETAIL_EXPECTED_HPP

namespace cublasdx {
    namespace detail {
        template<class T, class E>
        class expected {
        public:
            using value_type = T;
            using error_type = E;

        private:
            union storage_t {
                value_type value;
                error_type error;

                explicit storage_t(value_type const& input): value(input) {}
                explicit storage_t(value_type&& input): value(static_cast<value_type&&>(input)) {}
                explicit storage_t(error_type const& input): error(input) {}
                explicit storage_t(error_type&& input): error(static_cast<error_type&&>(input)) {}
                ~storage_t() {}
            };

            bool has_value_;
            storage_t storage_;

            void destroy() {
                if (has_value_) {
                    storage_.value.~value_type();
                } else {
                    storage_.error.~error_type();
                }
            }

        public:
            explicit expected(value_type const& value):
                has_value_(true),
                storage_(value) {}

            explicit expected(value_type&& value):
                has_value_(true),
                storage_(static_cast<value_type&&>(value)) {}

            explicit expected(error_type const& error):
                has_value_(false),
                storage_(error) {}

            explicit expected(error_type&& error):
                has_value_(false),
                storage_(static_cast<error_type&&>(error)) {}

            expected(expected const&) = delete;
            expected& operator=(expected const&) = delete;
            expected(expected&&) = delete;
            expected& operator=(expected&&) = delete;

            ~expected() {
                destroy();
            }

            explicit operator bool() const {
                return has_value();
            }

            bool has_value() const {
                return has_value_;
            }

            value_type& value() & {
                return storage_.value;
            }

            value_type const& value() const& {
                return storage_.value;
            }

            value_type&& value() && {
                return static_cast<value_type&&>(storage_.value);
            }

            value_type* operator->() {
                return &storage_.value;
            }

            value_type const* operator->() const {
                return &storage_.value;
            }

            value_type& operator*() & {
                return storage_.value;
            }

            value_type const& operator*() const& {
                return storage_.value;
            }

            value_type&& operator*() && {
                return static_cast<value_type&&>(storage_.value);
            }

            error_type& error() & {
                return storage_.error;
            }

            error_type const& error() const& {
                return storage_.error;
            }

            error_type&& error() && {
                return static_cast<error_type&&>(storage_.error);
            }
        };
    } // namespace detail
} // namespace cublasdx

#endif // CUBLASDX_DETAIL_EXPECTED_HPP
