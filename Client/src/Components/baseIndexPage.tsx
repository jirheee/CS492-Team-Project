import { Flex, HStack } from '@chakra-ui/layout';

const BaseIndexPage = ({ children }: any) => {
  return (
    <Flex
      w="full"
      h="wrap"
      my={30}
      flexDir="column"
      alignItems="center"
      justifyContent="center"
      bg="gray.50"
    >
      <HStack spacing={10} py={5}>
        {children}
      </HStack>
    </Flex>
  );
};

export default BaseIndexPage;
