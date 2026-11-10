import { useCallback, useEffect, useState } from "react";
import { getSearchType, setSearchType } from "@/utils/browserStorage";
import { DEFAULT_SEARCH_TYPE, isSearchTypeId, type SearchTypeId } from "@/app/(app)/search/searchTypes";
import type { Conversation } from "@/modules/searchHistory/conversations";

interface SearchTypeState {
  /** The type the next question runs on, and the one the control shows. */
  searchType: SearchTypeId;
  changeSearchType: (id: SearchTypeId) => void;
}

/**
 * The search type belongs to the conversation, not the page.
 *
 * As page state it applied to whichever conversation was open, so reopening a
 * Text conversation while Code was remembered sent the follow-up to the code
 * graph — which found nothing and, having no session turn to record, left a
 * hole in a conversation the pod otherwise holds in full. Now an open
 * conversation shows and uses its own type, the remembered preference applies
 * only to a new one, and picking the other type while a conversation is open
 * starts a new conversation rather than mixing the two in one.
 */
export default function useSearchType(
  instanceId: string | null,
  activeConvoId: string | null,
  activeConvo: Conversation | null,
  leaveConversation: () => void,
): SearchTypeState {
  const [preferred, setPreferred] = useState<SearchTypeId>(DEFAULT_SEARCH_TYPE);

  // Restored in an effect rather than a lazy useState initializer: this
  // component is prerendered on the server, where localStorage does not exist,
  // so reading it during render would hydrate a "Code" client over a "Text"
  // server tree. Re-runs on a workspace switch, which is what keys the choice
  // to the pod whose contents it describes.
  //
  // It cannot undo a click made before instanceId arrives: /search is in
  // CustomAppShell's POD_DEPENDENT_PATHS, so this page does not mount at all
  // until cogniInstance exists.
  useEffect(() => {
    if (!instanceId) return;
    const stored = getSearchType(instanceId);
    setPreferred(isSearchTypeId(stored) ? stored : DEFAULT_SEARCH_TYPE);
  }, [instanceId]);

  // An id remembered from before a reload, whose turns have not loaded yet, is
  // a pod session — and every conversation the pod holds is Text, since Code
  // turns are never recorded there.
  const searchType: SearchTypeId = activeConvoId ? (activeConvo?.searchType ?? "text") : preferred;

  const changeSearchType = useCallback(
    (id: SearchTypeId) => {
      setPreferred(id);
      if (instanceId) setSearchType(instanceId, id);
      if (activeConvoId && id !== searchType) leaveConversation();
    },
    [instanceId, activeConvoId, searchType, leaveConversation],
  );

  return { searchType, changeSearchType };
}
