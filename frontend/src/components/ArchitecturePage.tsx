import Box from "@cloudscape-design/components/box";
import ContentLayout from "@cloudscape-design/components/content-layout";
import Header from "@cloudscape-design/components/header";

export function ArchitecturePage() {
  return (
    <ContentLayout
      header={
        <Header variant="h1" description="System architecture diagram">
          Architecture
        </Header>
      }
    >
      <Box padding="l">
        <img
          src="/architecture.png"
          alt="System architecture diagram"
          style={{ maxWidth: "100%", height: "auto", display: "block" }}
        />
      </Box>
    </ContentLayout>
  );
}
