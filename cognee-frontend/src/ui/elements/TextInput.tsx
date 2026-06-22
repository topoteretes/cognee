import { Flex, TextInput, TextInputProps } from "@mantine/core";
import { forwardRef } from "react";

interface InputProps extends TextInputProps {
  label: string;
}

const CustomTextInput = forwardRef<HTMLInputElement, InputProps>(
  (props, ref) => {
    const { label, placeholder, ...rest } = props;

    return (
      <Flex className="flex-col gap-[8px]">
        <TextInput
          ref={ref}
          label={label}
          classNames={{
            input: "!h-[40px] !rounded-lg !border-[#E4E4E7] focus:!border-[#6510F4] focus:!border-2 focus:!shadow-[0_0_0_3px_rgba(101,16,244,0.1)]",
            label: "!text-sm mb-[8px] !font-normal !text-[#3F3F46]",
          }}
          placeholder={placeholder ?? label}
          {...rest}
        />
      </Flex>
    );
  },
);

CustomTextInput.displayName = "CustomTextInput";

export default CustomTextInput;
