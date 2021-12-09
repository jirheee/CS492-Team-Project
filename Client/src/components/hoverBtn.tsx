import { VStack, Heading, Text } from '@chakra-ui/layout';
import { useState } from 'react';

const HoverBtn = ({
  baseText,
  hoverText
}: {
  baseText: string;
  hoverText: string;
}) => {
  const [hover, setHover] = useState(false);
  const handleMouseOver = () => {
    setHover(true);
  };
  const handleMouseLeave = () => {
    setHover(false);
  };
  return (
    <VStack
      w="xs"
      h="xs"
      bg="gray.100"
      justify="center"
      onMouseOver={handleMouseOver}
      onMouseLeave={handleMouseLeave}
      _hover={{
        background: 'gray.200',
        color: 'teal.500'
      }}
    >
      <Heading textAlign="center">{baseText}</Heading>
      {hover ? <Text>{hoverText}</Text> : null}
    </VStack>
  );
};

export default HoverBtn;
