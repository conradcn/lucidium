import type { JSX } from "react";
import { InterviewStepShell } from "./InterviewStepShell";

interface Props {
  options: string[];
  onAnswer: (value: string, isFreeText: boolean) => void;
}

export function VisualStyleStep({ options, onAnswer }: Props): JSX.Element {
  return (
    <InterviewStepShell
      title="Visual style"
      description="The aesthetic the engine will draw in."
      options={options}
      onAnswer={onAnswer}
    />
  );
}
