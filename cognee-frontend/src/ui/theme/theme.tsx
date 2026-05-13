import { createTheme } from "@mantine/core";
import colors from "./colors";
import typography from "./typography";

const theme = createTheme({
  ...colors,
  ...typography,
  primaryColor: "primary2",
  defaultRadius: "0.5rem",
  cursorType: "pointer",
});

export default theme;
