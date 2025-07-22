> [!IMPORTANTE]
> **Nota para contribuidores:** Cuando crees una nueva rama, hazlo a partir de la rama `dev`.

# 🎉 ¡Bienvenido a **cognee**! 

¡Estamos emocionados de que estés interesado en contribuir a nuestro proyecto!  
Queremos asegurarnos de que cada usuario y colaborador se sienta bienvenido, incluido y con apoyo para participar en la comunidad de cognee.  
Esta guía te ayudará a comenzar y asegurará que tus contribuciones puedan integrarse de manera eficiente al proyecto.

## 🌟 Enlaces Rápidos

- [Código de Conducta](CODE_OF_CONDUCT.md)
- [Comunidad en Discord](https://discord.gg/bcy8xFAtfd)  
- [Seguimiento de Issues](https://github.com/topoteretes/cognee/issues)
- [Documentación de Cognee](https://docs.cognee.ai)

## 1. 🚀 Formas de Contribuir

Puedes contribuir a **cognee** de muchas maneras:

- 📝 Informar errores o sugerir nuevas funciones
- 💡 Mejorar la documentación
- 🔍 Revisar pull requests
- 🛠️ Contribuir con código o pruebas
- 🌐 Ayudar a otros usuarios

## 📫 Ponerse en Contacto

Hay varias formas de conectar con el equipo y la comunidad de **cognee**:

### Colaboración vía GitHub
- [Abrir un issue](https://github.com/topoteretes/cognee/issues) para errores, sugerencias o discusiones
- Enviar pull requests para contribuir con código o documentación
- Unirte a discusiones en issues y PRs existentes

### Canales Comunitarios
- Únete a nuestra [comunidad en Discord](https://discord.gg/bcy8xFAtfd) para charlas en tiempo real
- Participa en eventos y discusiones comunitarias
- Recibe ayuda de otros miembros

### Contacto Directo
- Correo electrónico: vasilije@cognee.ai  
- Para asuntos comerciales o sensibles, contacta por email  
- Para preguntas generales, prefiere canales públicos como GitHub o Discord

Nuestro objetivo es responder en un plazo de 2 días hábiles. Para respuestas más rápidas, usa nuestro canal de Discord donde toda la comunidad puede ayudarte.

## Etiquetas de Issues

Usamos las siguientes etiquetas para ayudarte a encontrar issues adecuados:

- `good first issue` - Perfecto para quienes contribuyen por primera vez
- `bug` - Algo no está funcionando correctamente
- `documentation` - Mejoras o añadidos a la documentación
- `enhancement` - Nuevas funcionalidades o mejoras
- `help wanted` - Se necesita atención o ayuda adicional
- `question` - Se requiere más información
- `wontfix` - No se trabajará en esto

¿Buscas por dónde empezar? Filtra por [good first issues](https://github.com/topoteretes/cognee/labels/good%20first%20issue).

---

## 2. 🛠️ Configuración del Entorno de Desarrollo

### Haz un Fork y Clona

1. Haz un fork del repositorio [**cognee**](https://github.com/topoteretes/cognee)  
2. Clona tu fork:
```bash
git clone https://github.com/<tu-usuario-de-github>/cognee.git
cd cognee
```

Si estás trabajando con Vector o Graph Adapters:

1. Haz un fork del repositorio [**cognee-community**](https://github.com/topoteretes/cognee-community)  
2. Clona tu fork:
```bash
git clone https://github.com/<tu-usuario-de-github>/cognee-community.git
cd cognee-community
```

### Crear una Rama

Crea una nueva rama para tu trabajo:
```bash
git checkout -b feature/nombre-de-tu-feature
```

---

## 3. 🎯 Haciendo Cambios

1. **Estilo de Código**: Sigue los estándares del proyecto  
2. **Documentación**: Actualiza la documentación relevante  
3. **Pruebas**: Agrega tests para nuevas funcionalidades  
4. **Commits**: Usa mensajes claros para los commits

### Ejecutar Pruebas
```bash
python cognee/cognee/tests/test_library.py
```

---

## 4. 📤 Enviar Cambios

1. Sube tus cambios:
```bash
git add .
git commit -s -m "Descripción de tus cambios"
git push origin feature/nombre-de-tu-feature
```

2. Crea un Pull Request:
   - Ve al repositorio [**cognee**](https://github.com/topoteretes/cognee)
   - Haz clic en "Compare & Pull Request" y abre el PR contra la rama `dev`
   - Completa el template del PR con los detalles de tus cambios

---

## 5. 📜 Certificado de Origen del Desarrollador (DCO)

Todas las contribuciones deben estar firmadas indicando conformidad con el DCO:

```bash
git config alias.cos "commit -s"  # Crear alias para commits firmados
```

Cuando tu PR esté listo, incluye esta declaración:
> "Afirmo que todo el código en cada commit de este pull request cumple con los términos del Certificado de Origen del Desarrollador de Topoteretes"

---

## 6. 🤝 Reglas de la Comunidad

- Sé respetuoso e inclusivo
- Ayuda a otros a aprender y crecer
- Sigue nuestro [Código de Conducta](CODE_OF_CONDUCT.md)
- Da retroalimentación constructiva
- Haz preguntas si tienes dudas

---

## 7. 📫 Obtener Ayuda

- Abre un [issue](https://github.com/topoteretes/cognee/issues)
- Únete a nuestra comunidad en Discord
- Revisa la documentación existente

---

¡Gracias por contribuir a **cognee**! 🌟