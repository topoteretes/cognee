{{/*
Expand the chart name.
*/}}
{{- define "cognee.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{/*
Create a default fully qualified app name.
*/}}
{{- define "cognee.fullname" -}}
{{- if .Values.fullnameOverride -}}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- .Release.Name | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- end -}}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "cognee.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{/*
Common labels.
*/}}
{{- define "cognee.labels" -}}
helm.sh/chart: {{ include "cognee.chart" . }}
app.kubernetes.io/name: {{ include "cognee.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end -}}

{{/*
Common selector labels.
*/}}
{{- define "cognee.selectorLabels" -}}
app.kubernetes.io/name: {{ include "cognee.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end -}}

{{- define "cognee.backendName" -}}
{{- printf "%s-cognee" (include "cognee.fullname" .) | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "cognee.postgresName" -}}
{{- printf "%s-postgres" (include "cognee.fullname" .) | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "cognee.configMapName" -}}
{{- printf "%s-config" (include "cognee.fullname" .) | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "cognee.persistenceClaimName" -}}
{{- if .Values.cognee.persistence.existingClaim -}}
{{- .Values.cognee.persistence.existingClaim -}}
{{- else -}}
{{- printf "%s-data" (include "cognee.fullname" .) | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- end -}}

{{- define "cognee.llmSecretName" -}}
{{- if .Values.cognee.secrets.existingSecret -}}
{{- .Values.cognee.secrets.existingSecret -}}
{{- else -}}
{{- printf "%s-llm" (include "cognee.fullname" .) | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- end -}}

{{- define "cognee.databaseSecretName" -}}
{{- if .Values.postgres.enabled -}}
{{- if .Values.postgres.auth.existingSecret -}}
{{- .Values.postgres.auth.existingSecret -}}
{{- else -}}
{{- printf "%s-postgres" (include "cognee.fullname" .) | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- else -}}
{{- required "externalDatabase.existingSecret is required when postgres.enabled=false" .Values.externalDatabase.existingSecret -}}
{{- end -}}
{{- end -}}

{{- define "cognee.generatedPostgresPassword" -}}
{{- $secretName := include "cognee.databaseSecretName" . -}}
{{- $passwordKey := .Values.postgres.auth.passwordKey -}}
{{- $existingSecret := lookup "v1" "Secret" .Release.Namespace $secretName -}}
{{- if and $existingSecret (hasKey $existingSecret.data $passwordKey) -}}
{{- index $existingSecret.data $passwordKey | b64dec -}}
{{- else -}}
{{- randAlphaNum 32 -}}
{{- end -}}
{{- end -}}

{{- define "cognee.databaseHost" -}}
{{- if .Values.postgres.enabled -}}
{{- include "cognee.postgresName" . -}}
{{- else -}}
{{- required "externalDatabase.host is required when postgres.enabled=false" .Values.externalDatabase.host -}}
{{- end -}}
{{- end -}}

{{- define "cognee.databasePort" -}}
{{- if .Values.postgres.enabled -}}
{{- .Values.postgres.service.port -}}
{{- else -}}
{{- .Values.externalDatabase.port -}}
{{- end -}}
{{- end -}}

{{- define "cognee.databaseUsernameKey" -}}
{{- if .Values.postgres.enabled -}}
{{- .Values.postgres.auth.usernameKey -}}
{{- else -}}
{{- .Values.externalDatabase.usernameKey -}}
{{- end -}}
{{- end -}}

{{- define "cognee.databasePasswordKey" -}}
{{- if .Values.postgres.enabled -}}
{{- .Values.postgres.auth.passwordKey -}}
{{- else -}}
{{- .Values.externalDatabase.passwordKey -}}
{{- end -}}
{{- end -}}

{{- define "cognee.databaseNameKey" -}}
{{- if .Values.postgres.enabled -}}
{{- .Values.postgres.auth.databaseKey -}}
{{- else -}}
{{- .Values.externalDatabase.databaseKey -}}
{{- end -}}
{{- end -}}

{{- define "cognee.serviceAccountName" -}}
{{- if .Values.cognee.serviceAccount.create -}}
{{- default (include "cognee.backendName" .) .Values.cognee.serviceAccount.name -}}
{{- else -}}
{{- default "default" .Values.cognee.serviceAccount.name -}}
{{- end -}}
{{- end -}}
