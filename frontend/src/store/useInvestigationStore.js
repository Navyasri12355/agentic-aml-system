import { create } from 'zustand';
import { investigateAccount } from '../api/client';

const useInvestigationStore = create((set) => ({
  accountId: '',
  selectedFile: null,
  isInvestigating: false,
  error: null,
  result: null,

  setAccountId: (id) => set({ accountId: id }),
  setSelectedFile: (file) => set({ selectedFile: file }),
  
  startInvestigation: async () => {
    set((state) => {
      if (!state.accountId) return state;
      return { isInvestigating: true, error: null, result: null };
    });

    try {
      const state = useInvestigationStore.getState();
      if (!state.accountId) return;

      const response = await investigateAccount(state.accountId, state.selectedFile);
      set({ result: response.data, isInvestigating: false });
    } catch (err) {
      set({ 
        error: err.response?.data?.detail || err.message || 'Investigation failed',
        isInvestigating: false 
      });
    }
  },

  reset: () => set({ accountId: '', selectedFile: null, isInvestigating: false, error: null, result: null })
}));

export default useInvestigationStore;
