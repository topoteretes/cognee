import { CustomAppShell } from "@/ui/layout";
import AppProvider from "@/modules/tenant/AppProvider";
import { NavbarProvider } from "@/ui/layout/NavbarContext";
import { FilterProvider } from "@/ui/layout/FilterContext";
import MobileMenuButton from "@/ui/layout/MobileMenuButton";

export default function AppLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <AppProvider>
      <FilterProvider>
        <NavbarProvider>
          <CustomAppShell>{children}</CustomAppShell>
          <MobileMenuButton />
        </NavbarProvider>
      </FilterProvider>
    </AppProvider>
  );
}
