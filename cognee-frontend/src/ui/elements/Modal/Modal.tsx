interface ModalProps {
  isOpen: boolean;
  children: React.ReactNode;
}

export default function Modal({ isOpen, children }: ModalProps) {
  return isOpen && (
    <div className="fixed top-0 left-0 right-0 bottom-0 backdrop-blur-lg z-20 flex items-center justify-center">
      {children}
    </div>
  );
}
