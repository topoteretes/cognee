import { useCallback, useEffect, useState } from 'react';
import {
  CTAButton,
  DropdownSelect,
  FormGroup,
  FormInput,
  FormLabel,
  Input,
  Spacer,
  Stack,
  useBoolean,
} from 'ohmy-ui';
import { LoadingIndicator } from '@/ui/App';
import { fetch } from '@/utils';

interface SelectOption {
  label: string;
  value: string;
}

interface SettingsForm extends HTMLFormElement {
  vectorDBUrl: HTMLInputElement;
  vectorDBApiKey: HTMLInputElement;
  llmApiKey: HTMLInputElement;
}

const defaultProvider = {
  label: 'OpenAI',
  value: 'openai',
};

const defaultModel = {
  label: 'gpt-4o-mini',
  value: 'gpt-4o-mini',
};

export default function Settings({ onDone = () => {}, submitButtonText = 'Save' }) {
  const [llmConfig, setLLMConfig] = useState<{
    apiKey: string;
    model: SelectOption;
    models: {
      [key: string]: SelectOption[];
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

  const saveConfig = (event: React.FormEvent<SettingsForm>) => {
    event.preventDefault();
    const formElements = event.currentTarget;

    const newVectorConfig = {
      provider: vectorDBConfig?.provider.value,
      url: formElements.vectorDBUrl.value,
      apiKey: formElements.vectorDBApiKey.value,
    };

    const newLLMConfig = {
      provider: llmConfig?.provider.value,
      model: llmConfig?.model.value,
      apiKey: formElements.llmApiKey.value,
    };

    startSaving();

    fetch('/v1/settings', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        llm: newLLMConfig,
        vectorDb: newVectorConfig,
      }),
    })
      .then(() => {
        onDone();
      })
      .finally(() => stopSaving());
  };

  const handleVectorDBChange = useCallback((newVectorDBProvider: SelectOption) => {
    setVectorDBConfig((config) => {
      if (config?.provider !== newVectorDBProvider) {
        return {
         ...config,
          options: config?.options || [],
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
          provider: newLLMProvider,
          providers: config?.providers || [],
          model: config?.models?.[newLLMProvider.value]?.[0] || defaultModel,
          models: config?.models || {},
          apiKey: config?.apiKey || '',
        };
      }
      return config;
    });
  }, []);

  const handleLLMModelChange = useCallback((newLLMModel: SelectOption) => {
    setLLMConfig((config) => {
      if (config?.model !== newLLMModel) {
        return {
          provider: config?.provider || defaultProvider,
          providers: config?.providers || [],
          model: newLLMModel,
          models: config?.models || {},
          apiKey: config?.apiKey || '',
        };
      }
      return config;
    });
  }, []);

  useEffect(() => {
    const fetchConfig = async () => {
      const response = await fetch('/v1/settings');
      const settings = await response.json();

      if (!settings.llm.model) {
        settings.llm.model = settings.llm.models[settings.llm.provider.value][0];
      }
      setLLMConfig(settings.llm);
      setVectorDBConfig(settings.vectorDb);
    };
    fetchConfig();
  }, []);

  return (
    <form onSubmit={saveConfig} style={{ width: '100%' }}>
      <Stack gap="4" orientation="vertical">
        <Stack gap="4" orientation="vertical">
          <FormGroup orientation="vertical" align="center/" gap="2">
            <FormLabel>LLM provider:</FormLabel>
            <DropdownSelect
              value={llmConfig?.provider || null}
              options={llmConfig?.providers || []}
              onChange={handleLLMProviderChange}
            />
          </FormGroup>
          <FormGroup orientation="vertical" align="center/" gap="2">
            <FormLabel>LLM model:</FormLabel>
            <DropdownSelect
              value={llmConfig?.model || null}
              options={llmConfig?.provider ? llmConfig?.models[llmConfig?.provider.value] : []}
              onChange={handleLLMModelChange}
            />
          </FormGroup>
          <FormInput>
            <Input defaultValue={llmConfig?.apiKey} name="llmApiKey" placeholder="LLM API key" />
          </FormInput>
        </Stack>

        <Stack gap="2" orientation="vertical">
          <FormGroup orientation="vertical" align="center/" gap="2">
            <FormLabel>Vector DB provider:</FormLabel>
            <DropdownSelect
              value={vectorDBConfig?.provider || null}
              options={vectorDBConfig?.options || []}
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
              <CTAButton type="submit">
                <Stack gap="2" orientation="vertical" align="center/">
                  {submitButtonText}
                  {isSaving && <LoadingIndicator />}
                </Stack>
              </CTAButton>
            </Spacer>
          </Stack>
        </Stack>
      </Stack>
    </form>
  )
}
