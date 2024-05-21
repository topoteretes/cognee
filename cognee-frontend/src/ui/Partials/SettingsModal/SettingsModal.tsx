import { CTAButton, DropdownSelect, FormGroup, FormInput, FormLabel, H2, H3, Input, Modal, Spacer, Stack, useBoolean } from 'ohmy-ui';
import { useCallback, useEffect, useState } from 'react';

interface SelectOption {
  label: string;
  value: string;
}

export default function SettingsModal({ isOpen = false, onClose = () => {} }) {
  const [llmConfig, setLLMConfig] = useState<{ openAIApiKey: string }>();
  const [vectorDBConfig, setVectorDBConfig] = useState<{
    choice: SelectOption;
    options: SelectOption[];
    url: string;
    apiKey: string;
  }>();

  const {
    value: isSaving,
    setTrue: startSaving,
    setFalse: stopSaving,
  } = useBoolean(false);

  const saveConfig = (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const newOpenAIApiKey = event.target.openAIApiKey.value;
    const newVectorDBChoice = vectorDBConfig?.choice.value;
    const newVectorDBUrl = event.target.vectorDBUrl.value;
    const newVectorDBApiKey = event.target.vectorDBApiKey.value;

    startSaving();

    fetch('http://0.0.0.0:8000/settings', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        llm: {
          openAIApiKey: newOpenAIApiKey,
        },
        vectorDB: {
          choice: newVectorDBChoice,
          url: newVectorDBUrl,
          apiKey: newVectorDBApiKey,
        },
      }),
    })
      .then(() => {
        onClose();
      })
      .finally(() => stopSaving());
  };

  const handleVectorDBChange = useCallback((newChoice: SelectOption) => {
    setVectorDBConfig((config) => {
      if (config?.choice !== newChoice) {
        return {
         ...config,
          choice: newChoice,
          url: '',
          apiKey: '',
        };
      }
      return config;
    });
  }, []);

  useEffect(() => {
    const fetchVectorDBChoices = async () => {
      const response = await fetch('http://0.0.0.0:8000/settings');
      const settings = await response.json();

      setLLMConfig(settings.llm);
      setVectorDBConfig(settings.vectorDB);
    };
    isOpen && fetchVectorDBChoices();
  }, [isOpen]);

  return (
    <Modal isOpen={isOpen} onClose={onClose}>
      <Stack gap="4" orientation="vertical" align="center/">
        <H2>Settings</H2>
        <form onSubmit={saveConfig} style={{ width: '100%' }}>
          <Stack gap="2" orientation="vertical">
            <H3>LLM Config</H3>
            <FormGroup orientation="vertical" align="center/" gap="1">
              <FormLabel>OpenAI API Key</FormLabel>
              <FormInput>
                <Input defaultValue={llmConfig?.openAIApiKey} name="openAIApiKey" placeholder="OpenAI API Key" />
              </FormInput>
            </FormGroup>

            <H3>Vector Database Config</H3>
            <DropdownSelect
              value={vectorDBConfig?.choice}
              options={vectorDBConfig?.options}
              onChange={handleVectorDBChange}
            />
            <FormGroup orientation="vertical" align="center/" gap="1">
              <FormLabel>Vector DB url</FormLabel>
              <FormInput>
                <Input defaultValue={vectorDBConfig?.url} name="vectorDBUrl" placeholder="Vector DB API url" />
              </FormInput>
            </FormGroup>
            <FormGroup orientation="vertical" align="center/" gap="1">
              <FormLabel>Vector DB API key</FormLabel>
              <FormInput>
                <Input defaultValue={vectorDBConfig?.apiKey} name="vectorDBApiKey" placeholder="Vector DB API key" />
              </FormInput>
            </FormGroup>
            <Stack align="/end">
              <Spacer top="2">
                <CTAButton type="submit">Save</CTAButton>
              </Spacer>
            </Stack>
          </Stack>
        </form>
      </Stack>
    </Modal>
  )
}
