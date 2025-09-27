import React from "react";

const StatusDot = ({ isActive, className }: { isActive: boolean, className?: string }) => {
  return (
    <span
      className={`inline-block w-3 h-3 rounded-full ${className} ${
        isActive ? "bg-green-500" : "bg-red-500"
      }`}
    />
  );
};

export default StatusDot;
