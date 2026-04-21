import { CustomAppShell } from "@/ui/Layout";
import AppProvider from "@/modules/tenant/AppProvider";
import { NavbarProvider } from "@/ui/Layout/NavbarContext";
import { FilterProvider } from "@/ui/Layout/FilterContext";
import MobileMenuButton from "@/ui/Layout/MobileMenuButton";

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
