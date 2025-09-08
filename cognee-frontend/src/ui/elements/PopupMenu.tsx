"use client";

import { useBoolean, useOutsideClick } from "@/utils";
import { MenuIcon } from "@/ui/Icons";
import { IconButton } from "@/ui/elements";
import classNames from 'classnames';

interface PopupMenuProps {
  children: React.ReactNode;
  triggerElement?: React.ReactNode;
  triggerClassName?: string;
  openToRight?: boolean;
}

export default function PopupMenu({ triggerElement, triggerClassName, children, openToRight = false }: PopupMenuProps) {
  const {
    value: isMenuOpen,
    setFalse: closeMenu,
    toggle: toggleMenu,
  } = useBoolean(false);

  const menuRootRef = useOutsideClick<HTMLDivElement>(closeMenu);

  return (
    <div className="relative inline-block" ref={menuRootRef}>
      <IconButton as="div" className={triggerClassName} onClick={toggleMenu}>
        {triggerElement || <MenuIcon />}
      </IconButton>

      {isMenuOpen && (
        <div
          className={
            classNames(
              "absolute top-full flex flex-col gap-4 pl-1 py-3 pr-4",
              "whitespace-nowrap bg-white border-1 border-gray-100 z-10",
              {
                "left-0": openToRight,
                "right-0": !openToRight,
              },
            )
          }
        >
          {children}
        </div>
      )}
    </div>
  );
};
