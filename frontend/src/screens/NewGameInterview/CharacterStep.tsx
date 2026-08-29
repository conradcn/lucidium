import type { JSX } from "react";
import { InterviewStepShell } from "./InterviewStepShell";

interface Props {
  options: string[];
  loading?: boolean;
  onAnswer: (value: string, isFreeText: boolean) => void;
}

export function CharacterStep({ options, loading, onAnswer }: Props): JSX.Element {
  return (
    <InterviewStepShell
      title="Your character"
      options={options}
      loading={loading ?? false}
      freeTextPlaceholder="Or describe your own..."
      onAnswer={onAnswer}
    />
  );
}
