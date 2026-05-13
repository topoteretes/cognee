import { Flex, PasswordInput, PasswordInputProps } from "@mantine/core";
import { forwardRef } from "react";

interface InputProps extends PasswordInputProps {
  label: string;
}

const CustomPasswordInput = forwardRef<HTMLInputElement, InputProps>(
  (props, ref) => {
    const { label, placeholder, ...rest } = props;

    return (
      <Flex className="flex-col gap-[8px]">
        <PasswordInput
          ref={ref}
          label={label}
          classNames={{
            input: "!h-[40px] !rounded-lg !border-[#E4E4E7] focus-within:!border-[#6510F4] focus-within:!border-2 focus-within:!shadow-[0_0_0_3px_rgba(101,16,244,0.1)]",
            label: "!text-sm mb-[8px] !font-normal !text-[#3F3F46]",
          }}
          placeholder={placeholder ?? label}
          {...rest}
        />
      </Flex>
    );
  },
);

CustomPasswordInput.displayName = "CustomPasswordInput";

export default CustomPasswordInput;
