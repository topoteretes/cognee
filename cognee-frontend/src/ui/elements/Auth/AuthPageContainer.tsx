import { PropsWithChildren } from "react";
import TetrisBackground from "./TetrisBackground";

// Auth pages share the landing hero's look: pure black with the 33px grid
// and the falling-tetromino canvas behind a centered content column.
export default function AuthPageContainer({ children }: PropsWithChildren) {
  return (
    <div
      className="relative h-screen overflow-hidden text-[#EDECEA]"
      style={{
        backgroundColor: "#000000",
        backgroundImage:
          "linear-gradient(rgba(244,244,244,0.10) 1px, transparent 1px), linear-gradient(90deg, rgba(244,244,244,0.10) 1px, transparent 1px)",
        backgroundSize: "33px 33px",
      }}
    >
      <TetrisBackground />
      <div className="relative z-[3] flex h-screen w-full flex-row">{children}</div>
    </div>
  );
}
