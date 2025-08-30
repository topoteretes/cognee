import { MenuIcon } from "@/ui/Icons";

export default function PopupMenu({ children  }: { children: React.ReactNode  }) {
  return (
    <details className="relative">
      <summary className="list-none">
        <div className="p-[0.5rem] m-[-0.5rem] cursor-pointer hover:bg-white rounded-xl">
          <MenuIcon />
        </div>
      </summary>

      <div className="absolute right-0 top-full flex flex-col gap-4 pl-1 py-3 pr-4 whitespace-nowrap bg-white border-1 border-gray-100 z-10">
        {children}
      </div>
    </details>
  );
};
