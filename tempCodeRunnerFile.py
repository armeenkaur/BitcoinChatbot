# Initialize Pinecone client
# pinecone_client = PineconeClient(
#     api_key=st.secrets["PINECONE_API_KEY"],
#     environment=st.secrets["PINECONE_ENVIRONMENT"]
# )

# index_name = 'my-index'
    
# try:
#     indexes = pinecone_client.list_indexes()
#     if index_name not in indexes:
#         pinecone_client.create_index(
#             name=index_name,
#             dimension=1536,
#             metric='euclidean',
#             spec=ServerlessSpec(
#                 cloud='aws', 
#                 region='us-east-1'
#             )
#         )
# except Exception as e:
#     print(f"An error occurred: {e}")

# # Connect to the Pinecone index
# index = pinecone_client.Index(index_name)

# # Update vectorstore to use PineconeVectorStore
# vectorstore = PineconeVectorStore(
#     index=index,
#     embedding_function=embeddings.embed_documents,
#     namespace="your_namespace"  # Optionally specify a namespace
# )