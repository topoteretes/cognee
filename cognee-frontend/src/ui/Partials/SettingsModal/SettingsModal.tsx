import { CTAButton, DropdownSelect, FormGroup, FormInput, FormLabel, H2, H3, Input, Modal, Spacer, Stack, useBoolean } from 'ohmy-ui';
import { useCallback, useEffect, useState } from 'react';

interface SelectOption {
  label: string;
  value: string;
}

export default function SettingsModal({ isOpen = false, onClose = () => {} }) {
  const [llmConfig, setLLMConfig] = useState<{
    apiKey: string;
    model: SelectOption;
    models: {
      openai: SelectOption[];
      ollama: SelectOption[];
      anthropic: SelectOption[];
    };
    provider: SelectOption;
    providers: SelectOption[];
  }>();
  const [vectorDBConfig, setVectorDBConfig] = useState<{
    url: string;
    apiKey: string;
    provider: SelectOption;
    options: SelectOption[];
  }>();

  const {
    value: isSaving,
    setTrue: startSaving,
    setFalse: stopSaving,
  } = useBoolean(false);

  const saveConfig = (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();

    const newVectorConfig = {
      provider: vectorDBConfig?.provider.value,
      url: event.target.vectorDBUrl.value,
      apiKey: event.target.vectorDBApiKey.value,
    };

    const newLLMConfig = {
      provider: llmConfig?.provider.value,
      model: llmConfig?.model.value,
      apiKey: event.target.llmApiKey.value,
    };

    startSaving();

    fetch('http://0.0.0.0:8000/settings', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        llm: newLLMConfig,
        vectorDB: newVectorConfig,
      }),
    })
      .then(() => {
        onClose();
      })
      .finally(() => stopSaving());
  };

  const handleVectorDBChange = useCallback((newVectorDBProvider: SelectOption) => {
    setVectorDBConfig((config) => {
      if (config?.provider !== newVectorDBProvider) {
        return {
         ...config,
          provider: newVectorDBProvider,
          url: '',
          apiKey: '',
        };
      }
      return config;
    });
  }, []);

  const handleLLMProviderChange = useCallback((newLLMProvider: SelectOption) => {
    setLLMConfig((config) => {
      if (config?.provider !== newLLMProvider) {
        return {
         ...config,
          provider: newLLMProvider,
          model: config?.models[newLLMProvider.value][0],
          apiKey: '',
        };
      }
      return config;
    });
  }, []);

  const handleLLMModelChange = useCallback((newLLMModel: SelectOption) => {
    setLLMConfig((config) => {
      if (config?.model !== newLLMModel) {
        return {
         ...config,
          model: newLLMModel,
        };
      }
      return config;
    });
  }, []);

  useEffect(() => {
    const fetchVectorDBChoices = async () => {
      const response = await fetch('http://0.0.0.0:8000/settings');
      const settings = await response.json();

      if (!settings.llm.model) {
        settings.llm.model = settings.llm.models[settings.llm.provider.value][0];
      }
      setLLMConfig(settings.llm);
      setVectorDBConfig(settings.vectorDB);
    };
    isOpen && fetchVectorDBChoices();
  }, [isOpen]);

  return (
    <Modal isOpen={isOpen} onClose={onClose}>
      <Stack gap="8" orientation="vertical" align="center/">
        <H2>Settings</H2>
        <form onSubmit={saveConfig} style={{ width: '100%' }}>
          <Stack gap="4" orientation="vertical">
            <Stack gap="2" orientation="vertical">
              <H3>LLM Config</H3>
              <FormGroup orientation="horizontal" align="center/" gap="4">
                <FormLabel>LLM provider:</FormLabel>
                <DropdownSelect
                  value={llmConfig?.provider}
                  options={llmConfig?.providers}
                  onChange={handleLLMProviderChange}
                />
              </FormGroup>
              <FormGroup orientation="horizontal" align="center/" gap="4">
                <FormLabel>LLM model:</FormLabel>
                <DropdownSelect
                  value={llmConfig?.model}
                  options={llmConfig?.provider ? llmConfig?.models[llmConfig?.provider.value] : []}
                  onChange={handleLLMModelChange}
                />
              </FormGroup>
              <FormInput>
                <Input defaultValue={llmConfig?.apiKey} name="llmApiKey" placeholder="LLM API key" />
              </FormInput>
            </Stack>

            <Stack gap="2" orientation="vertical">
              <H3>Vector Database Config</H3>
              <FormGroup orientation="horizontal" align="center/" gap="4">
                <FormLabel>Vector DB provider:</FormLabel>
                <DropdownSelect
                  value={vectorDBConfig?.provider}
                  options={vectorDBConfig?.options}
                  onChange={handleVectorDBChange}
                />
              </FormGroup>
              <FormInput>
                <Input defaultValue={vectorDBConfig?.url} name="vectorDBUrl" placeholder="Vector DB instance url" />
              </FormInput>
              <FormInput>
                <Input defaultValue={vectorDBConfig?.apiKey} name="vectorDBApiKey" placeholder="Vector DB API key" />
              </FormInput>
              <Stack align="/end">
                <Spacer top="2">
                  <CTAButton type="submit">Save</CTAButton>
                </Spacer>
              </Stack>
            </Stack>
          </Stack>
        </form>
      </Stack>
    </Modal>
  )
}
