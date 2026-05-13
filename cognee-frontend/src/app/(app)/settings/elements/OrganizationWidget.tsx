"use client";

import { Select, Stack, Text, Title } from "@mantine/core";
import { useTenant } from "@/modules/tenant/TenantProvider";
import { tokens } from "@/ui/theme/tokens";

export default function OrganizationWidget() {
  const { tenant, availableTenants, switchTenant } = useTenant();

  const data = availableTenants.map((t) => ({
    value: t.id,
    label: t.name,
  }));

  return (
    <Stack
      className="rounded-[0.5rem] px-[2rem] pt-[1.5rem] pb-[1.75rem] !gap-[0] min-w-[25rem] max-w-[29.5rem]"
      bg="white"
    >
      <Stack className="!gap-[0] mb-[1.375rem]">
        <Title size="h2" mb="0.125rem">
          Organization
        </Title>
        <Text c={tokens.textMuted} size="lg">
          Switch between organizations you belong to
        </Text>
      </Stack>
      <Select
        label="Active organization"
        placeholder="Select organization"
        data={data}
        value={tenant?.tenant_id ?? null}
        onChange={(val) => val && switchTenant(val)}
        radius="0.5rem"
        classNames={{ input: "!h-[2.75rem] !border-cognee-border" }}
      />
    </Stack>
  );
}
