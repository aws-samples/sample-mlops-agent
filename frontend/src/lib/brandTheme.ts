import { applyTheme, type Theme } from "@cloudscape-design/components/theming";

// ── Sample MLOps Agent brand palette — "Lab Instrument" ─────────────────────
// A precise, scientific-instrument identity built around a deep teal-green
// (the flask/chemistry mark), distinct from stock AWS blue, with a warm amber
// accent for highlights. These same hex values are mirrored in the Cognito
// Hosted UI CSS (cdk .../cognito-hosted-ui.css) so login and app read as one
// product. Keep the two in sync when changing.
export const BRAND = {
  primary: "#0b7268", // deep teal-green (flask)
  primaryHover: "#095f57",
  primaryActive: "#074c45",
  accent: "#c8781e", // amber — links / interactive text
  accentHover: "#a5631690",
  headerBg: "#08201e", // near-black teal for the app header band
  layoutBg: "#f4f7f6", // faint cool-neutral canvas
} as const;

const brandTheme: Theme = {
  tokens: {
    // Primary action buttons → brand teal (was AWS blue).
    colorBackgroundButtonPrimaryDefault: BRAND.primary,
    colorBackgroundButtonPrimaryHover: BRAND.primaryHover,
    colorBackgroundButtonPrimaryActive: BRAND.primaryActive,
    // Interactive text (links, active nav) → amber accent, readable on light.
    colorTextInteractiveDefault: { light: BRAND.accent, dark: "#e2a34e" },
    colorTextInteractiveHover: { light: BRAND.primaryHover, dark: "#f0c07a" },
    colorTextAccent: { light: BRAND.primary, dark: "#3fb3a5" },
    // Subtle branded canvas + rounded containers for a softer, instrument feel.
    colorBackgroundLayoutMain: { light: BRAND.layoutBg, dark: "#0b1a19" },
    borderRadiusButton: "8px",
    borderRadiusContainer: "12px",
    // IBM Plex Sans — an engineered, technical-heritage typeface that suits the
    // instrument identity without the generic Inter/Roboto/Arial look. Loaded
    // via index.html; falls back to the platform UI stack if the webfont fails.
    fontFamilyBase:
      "'IBM Plex Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
  },
};

// Apply once at startup; returns a reset() we don't need (SPA lifetime).
export function applyBrandTheme(): void {
  applyTheme({ theme: brandTheme });
}
