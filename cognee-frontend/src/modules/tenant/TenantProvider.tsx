/**
 * Open-source stub — re-exports hooks from TenantContext.
 *
 * The SaaS version contains the full TenantProvider with Auth0/Stripe
 * initialization. ~15 files import useCogniInstance from this path,
 * so this stub ensures they resolve correctly.
 */
export { useTenant, useCogniInstance } from "./TenantContext";
