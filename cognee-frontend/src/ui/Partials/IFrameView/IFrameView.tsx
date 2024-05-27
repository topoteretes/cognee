interface IFrameViewProps {
  src: string;
}

export default function IFrameView({ src }: IFrameViewProps) {
  return (
    <iframe
      src={src}
      width="100%"
      height="100%"
    />
  );
}
