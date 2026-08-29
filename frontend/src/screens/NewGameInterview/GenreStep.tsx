import type { JSX } from "react";
import { InterviewStepShell } from "./InterviewStepShell";

interface Props {
  options: string[];
  onAnswer: (value: string, isFreeText: boolean) => void;
}

export function GenreStep({ options, onAnswer }: Props): JSX.Element {
  return (
    <InterviewStepShell title="Genre" options={options} onAnswer={onAnswer} compact />
  );
}
