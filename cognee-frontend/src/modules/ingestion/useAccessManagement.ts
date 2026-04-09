import { useCallback, useRef, useState } from 'react';
import { CogneeInstance } from '@/modules/instances/types';
import fetchTenants from '../users/fetchTenants';
import fetchUsers from '../users/fetchUsers';

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
    const [managedUsers, setManagedUsers] = useState<ManagedUser[]>([
        {id: '1', email: 'john.doe@unknown.com', roles: [{id: '1', name: 'Admin'}]},
        {id: '2', email: 'jane.smith@unknown.com', roles: [{id: '1', name: 'Admin'}]},
        {id: '3', email: 'bob.ross@art.com', roles: [{id: '1', name: 'Admin'}]}
    ]);

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
        });;
    }, []);

    const getManagedUsers = useCallback((tenantId: string) => {
        if (!cogniInstance) return Promise.resolve([]);

        return fetchUsers(tenantId, cogniInstance).then((response) => {
            allManagedUsers.current = response;
            setManagedUsers(response);
            return response;
        }).catch((error) => {
            console.error("Error fetching users: ", error.detail || error.message);
            throw error;
        });;
    }, [cogniInstance]);

  return { tenantId, tenants, managedUsers, getTenants, getManagedUsers };
};

export default useAccessManagement;
