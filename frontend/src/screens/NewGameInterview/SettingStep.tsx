import type { JSX } from "react";
import { InterviewStepShell } from "./InterviewStepShell";

interface Props {
  options: string[];
  onAnswer: (value: string, isFreeText: boolean) => void;
}

export function SettingStep({ options, onAnswer }: Props): JSX.Element {
  return (
    <InterviewStepShell
      title="Setting"
      description="Pick the world this story unfolds in, or describe your own."
      options={options}
      onAnswer={onAnswer}
    />
  );
}
