import { useCallback, useRef, useState } from 'react';
import { CogneeInstance } from '@/modules/instances/types';
import fetchTenants from '../users/fetchTenants';
import fetchUsers from '../users/fetchUsers';
import addUserToTenant from '../users/addUserToTenant';

interface Tenant {
  id: string;
  name: string;
}

interface ManagedUser {
    id: string;
    email: string;
    roles: Role[]
}

interface Role {
    id: string;
    name: string;
}

const useAccessManagement = (cogniInstance?: CogneeInstance | null) => {
    const allTenants = useRef<Tenant[]>([]);
    const [tenantId, setTenantId] = useState<string>();
    const [tenants, setTenants] = useState<Tenant[]>([{id: '1', name: "Cognee's organization"}])

    const allManagedUsers = useRef<ManagedUser[]>([])
    const [managedUsers, setManagedUsers] = useState<ManagedUser[]>([]);

    const getTenants = useCallback(() => {
        return fetchTenants().then((response) => {
            allTenants.current = response;
            // TODO: Current UI only supports one organization/tenant per logged in user
            setTenantId(response[0].id);
            setTenants(response);
            return response;
        }).catch((error) => {
            console.error("Error fetching tenants: ", error.detail || error.message);
            throw error;
        });
    }, []);

    const getManagedUsers = useCallback((tenantId: string) => {
        return fetchUsers(tenantId).then((response) => {
            const users = Array.isArray(response) ? response : [];
            allManagedUsers.current = users;
            setManagedUsers(users);
            return users;
        }).catch(() => {
            // Gracefully handle 403 etc. (e.g. guest on another tenant)
            setManagedUsers([]);
            return [];
        });
    }, []);

    const addUser = useCallback((email: string, tenantId: string) => {
        return addUserToTenant(email, tenantId).catch(() => {});
    }, []);

  return { tenantId, tenants, managedUsers, getTenants, getManagedUsers, addUser };
};

export default useAccessManagement;
